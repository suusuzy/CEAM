import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.GATLayer import GATv2Conv
from einops import rearrange
# from src.GCN import GCNConv
from src.multi_attention_forward import multi_head_attention_forward
from torch_geometric.utils import remove_self_loops



def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, batch_first=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.batch_first = batch_first

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        # Support loading checkpoints without batch_first
        if 'batch_first' not in state:
            state['batch_first'] = False

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
         - Inputs (when batch_first=False):
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs :
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.

        - Inputs (when batch_first=True):
        - query: :math:`(N, L, E)` where N is the batch size, L is the target sequence length, E is
          the embedding dimension.
        - key: :math:`(N, S, E)`, where N is the batch size, S is the source sequence length, E is
          the embedding dimension.
        - value: :math:`(N, S, E)` where N is the batch size, S is the source sequence length, E is
          the embedding dimension.
        - Outputs :
        - attn_output: :math:`(N, L, E)` where N is the batch size, L is the target sequence length,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        # 处理batch_first参数
        if self.batch_first:
            # 将输入从(N, L, E)转换为(L, N, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # if not self._qkv_same_embed_dim:
        #     return multi_head_attention_forward(
        #         query, key, value, self.embed_dim, self.num_heads,
        #         self.in_proj_weight, self.in_proj_bias,
        #         self.bias_k, self.bias_v, self.add_zero_attn,
        #         self.dropout, self.out_proj.weight, self.out_proj.bias,
        #         training=self.training,
        #         key_padding_mask=key_padding_mask, need_weights=need_weights,
        #         attn_mask=attn_mask, use_separate_proj_weight=True,
        #         q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
        #         v_proj_weight=self.v_proj_weight)
        # else:
        #     return multi_head_attention_forward(
        #         query, key, value, self.embed_dim, self.num_heads,
        #         self.in_proj_weight, self.in_proj_bias,
        #         self.bias_k, self.bias_v, self.add_zero_attn,
        #         self.dropout, self.out_proj.weight, self.out_proj.bias,
        #         training=self.training,
        #         key_padding_mask=key_padding_mask, need_weights=need_weights,
        #         attn_mask=attn_mask)
        # 调用原有的multi_head_attention_forward函数
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

        # 如果batch_first=True，将输出从(L, N, E)转换回(N, L, E)
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        atts = []

        for i in range(self.num_layers):
            output, attn = self.layers[i](output, src_mask=mask,
                                          src_key_padding_mask=src_key_padding_mask)
            atts.append(attn)
        if self.norm:
            output = self.norm(output)

        return output


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, mask):
        n_mask = mask + torch.eye(mask.shape[0], mask.shape[0]).cuda()
        n_mask = n_mask.float().masked_fill(n_mask == 0., float(-1e20)).masked_fill(n_mask == 1., float(0.0))
        output = self.transformer_encoder(src, mask=n_mask)
        return output


class DYNA(torch.nn.Module):

    def __init__(self, args, dropout_prob=0):
        super(DYNA, self).__init__()
        # 传参给GAT
        # self.gatoutputs = None  # 用于存储 outputs
        # self.gatnei_list = None  # 用于存储 nei_list
        # self.gatnode_abs = None  # 用于存储 node_abs

        # set parameters for network architecture
        self.embedding_size = [32]
        self.output_size = 2
        self.dropout_prob = dropout_prob
        self.args = args

        self.temporal_encoder_layer = TransformerEncoderLayer(d_model=32, nhead=8)

        emsize = 32  # embedding dimension
        nhid = 2048  # the dimension of the feedforward network model in TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 8  # the number of heads in the multihead-attention models
        dropout = 0.1  # the dropout value #0.1

        self.spatial_encoder_1 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
        self.spatial_encoder_2 = TransformerModel(emsize, nhead, nhid, nlayers, dropout)

        self.temporal_encoder_1 = TransformerEncoder(self.temporal_encoder_layer, 1)
        self.temporal_encoder_2 = TransformerEncoder(self.temporal_encoder_layer, 1)

        # Linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.input_embedding_layer_spatial = nn.Linear(2, 32)

        # Linear layer to output and fusion
        self.output_layer = nn.Linear(48, 3)  # Used for learning the positive definite matrix
        self.fusion_layer = nn.Linear(64, 32)

        # ReLU and dropout init修改了inplace操作-250214
        # remove inplace to checkinplace=False, inplace=False, inplace=False
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(self.dropout_prob)
        self.dropout_in2 = nn.Dropout(self.dropout_prob)

    def get_st_ed(self, batch_num):
        """
        :param batch_num: contains number of pedestrians in different scenes for a batch
        :type batch_num: list
        :return: st_ed: list of tuple contains start index and end index of pedestrians in different scenes
        :rtype: list
        """
        cumsum = torch.cumsum(batch_num, dim=0)
        st_ed = []
        for idx in range(1, cumsum.shape[0]):
            st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

        st_ed.insert(0, (0, int(cumsum[0])))

        return st_ed

    def get_node_index(self, seq_list):
        """

        :param seq_list: mask indicates whether pedestrain exists
        :type seq_list: numpy array [F, N], F: number of frames. N: Number of pedestrians (a mask to indicate whether
                                                                                            the pedestrian exists)
        :return: All the pedestrians who exist from the beginning to current frame
        :rtype: numpy array
        """
        for idx, framenum in enumerate(seq_list):

            if idx == 0:
                node_indices = framenum > 0
            else:
                node_indices *= (framenum > 0)

        return node_indices

    def update_batch_pednum(self, batch_pednum, ped_list):
        """

        :param batch_pednum: batch_num: contains number of pedestrians in different scenes for a batch
        :type list
        :param ped_list: mask indicates whether the pedestrian exists through the time window to current frame
        :type tensor
        :return: batch_pednum: contains number of pedestrians in different scenes for a batch after removing pedestrian who disappeared
        :rtype: list
        """
        updated_batch_pednum_ = copy.deepcopy(batch_pednum).cpu().numpy()
        updated_batch_pednum = copy.deepcopy(batch_pednum)

        cumsum = np.cumsum(updated_batch_pednum_)
        new_ped = copy.deepcopy(ped_list).cpu().numpy()

        for idx, num in enumerate(cumsum):
            num = int(num)
            if idx == 0:
                updated_batch_pednum[idx] = len(np.where(new_ped[0:num] == 1)[0])
            else:
                updated_batch_pednum[idx] = len(np.where(new_ped[int(cumsum[idx - 1]):num] == 1)[0])

        return updated_batch_pednum

    def mean_normalize_abs_input(self, node_abs, st_ed):
        """

        :param node_abs: Absolute coordinates of pedestrians
        :type Tensor
        :param st_ed: list of tuple indicates the indices of pedestrians belonging to the same scene
        :type List of tupule
        :return: node_abs: Normalized absolute coordinates of pedestrians
        :rtype: Tensor
        """
        node_abs = node_abs.permute(1, 0, 2)
        for st, ed in st_ed:
            mean_x = torch.mean(node_abs[st:ed, :, 0])
            mean_y = torch.mean(node_abs[st:ed, :, 1])

            node_abs[st:ed, :, 0] = (node_abs[st:ed, :, 0] - mean_x)
            node_abs[st:ed, :, 1] = (node_abs[st:ed, :, 1] - mean_y)

        return node_abs.permute(1, 0, 2)

    def get_predicted_velocity_and_nextstep(self, outputs_current, nodes_abs, nodes_norm):
        ped_num = outputs_current.shape[0]
        sigma = 10 ** (-8)
        sigma_ = torch.ones(ped_num, 2).cuda() * sigma
        Sigma = torch.diag_embed(sigma_)
        A = torch.zeros(ped_num, 2, 2).cuda()
        A[:, 0, 0] = outputs_current[:, 0]
        A[:, 1, 0] = outputs_current[:, 1]
        A[:, 1, 1] = outputs_current[:, 2]
        A_T = A.transpose(1, 2)
        pd_matrix = torch.bmm(A, A_T) + Sigma
        attractor_grad_func = lambda y: F.normalize(y)
        attractor_grad = -attractor_grad_func(nodes_norm)
        predicted_velocity = torch.bmm(pd_matrix, attractor_grad.unsqueeze(2)).squeeze()
        next_step = nodes_abs + 0.4 * predicted_velocity
        return predicted_velocity, next_step

    def forward(self, inputs, iftest=False):
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        num_Ped = nodes_norm.shape[1]

        outputs = torch.zeros(nodes_norm.shape[0], num_Ped, 2).cuda()
        GM = torch.zeros(nodes_norm.shape[0], num_Ped, 32).cuda()

        noise = get_noise((1, 16), 'gaussian')

        for framenum in range(self.args.seq_length - 1):
            if framenum >= self.args.obs_length and iftest:
                node_index = self.get_node_index(seq_list[:self.args.obs_length])
                updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                st_ed = self.get_st_ed(updated_batch_pednum)

                nodes_current = outputs[self.args.obs_length - 1:framenum, node_index] - shift_value[
                                                                                         self.args.obs_length:framenum + 1,
                                                                                         node_index]
                nodes_current = torch.cat((nodes_norm[:self.args.obs_length, node_index], nodes_current))
                node_abs_base = nodes_abs[:self.args.obs_length, node_index]

                node_abs_pred = outputs[self.args.obs_length - 1:framenum, node_index]
                node_abs = torch.cat((node_abs_base, node_abs_pred), dim=0)
                node_abs = self.mean_normalize_abs_input(node_abs, st_ed)
            else:
                node_index = self.get_node_index(seq_list[:framenum + 1])
                nei_list = nei_lists[framenum, node_index, :]
                nei_list = nei_list[:, node_index]
                updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                st_ed = self.get_st_ed(updated_batch_pednum)
                nodes_current = nodes_norm[:framenum + 1, node_index]
                node_abs = self.mean_normalize_abs_input(nodes_abs[:framenum + 1, node_index], st_ed)

            # Input Embedding
            if framenum == 0:
                temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
            else:
                temporal_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_temporal(nodes_current)))
                temporal_input_embedded = torch.cat(
                    [GM[:framenum, node_index], temporal_input_embedded[framenum:]], dim=0
                )

            spatial_input_embedded_ = self.dropout_in2(self.relu(self.input_embedding_layer_spatial(node_abs)))
            spatial_input_embedded = self.spatial_encoder_1(spatial_input_embedded_[-1].unsqueeze(1), nei_list)
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)[-1]
            temporal_input_embedded_last = self.temporal_encoder_1(temporal_input_embedded)[-1]
            temporal_input_embedded = temporal_input_embedded[:-1]

            fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=1)
            fusion_feat = self.fusion_layer(fusion_feat)

            spatial_input_embedded = self.spatial_encoder_2(fusion_feat.unsqueeze(1), nei_list)
            spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)

            temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=0)
            temporal_input_embedded = self.temporal_encoder_2(temporal_input_embedded)[-1]

            noise_to_cat = noise.repeat(temporal_input_embedded.shape[0], 1)
            temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded, noise_to_cat), dim=1)
            outputs_current = self.output_layer(temporal_input_embedded_wnoise)

            if framenum >= self.args.obs_length and iftest:
                curr_pred_velocity, predicted_nextstep = self.get_predicted_velocity_and_nextstep(outputs_current,
                                                                                                  outputs[
                                                                                                      framenum - 1, node_index],
                                                                                                  nodes_current[
                                                                                                      -1, node_index])
            else:
                curr_pred_velocity, predicted_nextstep = self.get_predicted_velocity_and_nextstep(outputs_current,
                                                                                                  nodes_abs[
                                                                                                      framenum, node_index],
                                                                                                  nodes_norm[
                                                                                                      framenum, node_index])

            outputs = torch.cat(
                [outputs[:framenum], predicted_nextstep.unsqueeze(0), outputs[framenum + 1:]], dim=0
            )
            GM = torch.cat(
                [GM[:framenum], temporal_input_embedded.unsqueeze(0), GM[framenum + 1:]], dim=0
            )
        # self.gatnei_list = nei_list
        # self.gatnode_abs = node_abs

        return outputs


class EnhancedCrossAttention(nn.Module):
    def __init__(self, d_model=64, nhead=8, dropout=0.1, use_relative_pos=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.use_relative_pos = use_relative_pos

        # 输入特征投影 - 使用更好的初始化
        self.dyna_proj = self._make_projection(2, d_model)
        self.gat_proj = self._make_projection(2, d_model)

        # 位置编码（如果启用）
        if use_relative_pos:
            self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=1000)

        # 交叉注意力层 - 使用batch_first=True简化维度处理
        self.cross_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True  # 支持[B,T,C]格式
        )

        # 自注意力层（增强时序建模能力）
        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # 改进的前馈网络
        self.ffn = self._make_ffn(d_model, dropout)

        # 输出处理 - 添加激活函数
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 添加特征归一化
        self.dyna_norm = nn.LayerNorm(d_model)
        self.gat_norm = nn.LayerNorm(d_model)
        
        # 改进的融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 参数初始化
        self._init_weights()

    def _make_projection(self, input_dim, output_dim):
        """创建特征投影层"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )

    def _make_ffn(self, d_model, dropout):
        """创建前馈网络"""
        return nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, dyna_out, gat_out, mask=None):
        """
        Args:
            dyna_out: [B, T, N, 2] 批次×时间步×节点数×特征 或 [T, N, 2]
            gat_out: [B, T, N, 2] 或 [T, N, 2]
            mask: [B, T] 可选的padding mask
        Returns:
            [B, T, N, 2] 或 [T, N, 2] 融合后的轨迹
        """
        original_shape = dyna_out.shape

        # 处理输入维度
        if len(original_shape) == 3:  # [T, N, 2]
            dyna_out = dyna_out.unsqueeze(0)  # [1, T, N, 2]
            gat_out = gat_out.unsqueeze(0)
            batch_added = True
        else:
            batch_added = False

        B, T, N, _ = dyna_out.shape

        # 重塑为[B*N, T, 2]进行处理
        dyna_reshaped = dyna_out.view(B * N, T, 2)
        gat_reshaped = gat_out.view(B * N, T, 2)

        # 1. 特征投影
        dyna_emb = self.dyna_proj(dyna_reshaped)  # [B*N, T, d_model]
        gat_emb = self.gat_proj(gat_reshaped)  # [B*N, T, d_model]

        # 2. 位置编码
        if self.use_relative_pos:
            dyna_emb = self.pos_encoding(dyna_emb)
            gat_emb = self.pos_encoding(gat_emb)

        # 3. 交叉注意力 (GAT指导Dynamic)
        cross_attn_out, _ = self.cross_attn(
            query=dyna_emb,
            key=gat_emb,
            value=gat_emb,
            key_padding_mask=None,  # 可以根据需要添加mask
            need_weights=False
        )  # [B*N, T, d_model]

        # 4. 残差连接+层归一化
        cross_output = self.norm1(dyna_emb + self.dropout(cross_attn_out))

        # 5. 自注意力（增强时序建模）
        self_attn_out, _ = self.self_attn(
            query=cross_output,
            key=cross_output,
            value=cross_output,
            need_weights=False
        )

        # 6. 残差连接+层归一化
        self_output = self.norm2(cross_output + self.dropout(self_attn_out))

        # 7. 前馈网络
        ffn_output = self.ffn(self_output)
        final_emb = self.norm3(self_output + ffn_output)  # [B*N, T, d_model]

        # 8. 智能融合门控
        # 将原始特征和注意力特征拼接
        concat_features = torch.cat([dyna_emb, final_emb], dim=-1)  # [B*N, T, 2*d_model]
        gate_weights = self.fusion_gate(concat_features)  # [B*N, T, d_model]

        # 应用门控
        fused_features = gate_weights * final_emb + (1 - gate_weights) * dyna_emb

        # 9. 输出投影
        pred_output = self.out_proj(fused_features)  # [B*N, T, 2]

        # 10. 与原始动态预测进行残差连接
        final_output = pred_output + dyna_reshaped

        # 恢复原始形状
        final_output = final_output.view(B, T, N, 2)

        # 如果原来是3D，移除batch维度
        if batch_added:
            final_output = final_output.squeeze(0)

        return final_output


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        seq_len = x.size(1)
        pos_emb = self.pe[:seq_len, :].transpose(0, 1)  # [1, T, d_model]
        return self.dropout(x + pos_emb.expand_as(x))

    # def get_attention_maps(self, dyna_out, gat_out):
    #     """可视化注意力权重"""
    #     dyna_emb = self.dyna_proj(dyna_out).permute(1, 0, 2)
    #     gat_emb = self.gat_proj(gat_out).permute(1, 0, 2)

    #     _, attn_weights = self.cross_attn(
    #         query=dyna_emb,
    #         key=gat_emb,
    #         value=gat_emb,
    #         average_attn_weights=False
    #     )
    #     return attn_weights  # [N_heads, T_query, T_key]



class CEAM(nn.Module):
    def __init__(self, args, dropout_prob=0):
        super(CEAM, self).__init__()
        self.args = args
        self.dropout_prob = dropout_prob
        self.K = 1

        # DYNA components
        self.dyna = DYNA(args, dropout_prob)
        # self.residual_0 = Residual_0(2, 4, use_1x1conv=True,)
        self.conv0 = nn.Conv2d(2, 4, kernel_size=3, padding=1, stride=1)
        self.BN0 = nn.BatchNorm2d(4, eps=1e-5, affine=True)
        self.dropout0 = nn.Dropout(0.3)

        # GAT components
        # self.gcn = GCNConv(in_channels=embed_size, out_channels=embed_size, K=self.K)
        self.gat = GATv2Conv(2, 8, add_self_loops=True)  # [4*4=16]
        # Reduce GAT output dimension from 256 to 152 to 32
        # self.residual = Residual(64,256,use_1x1conv= True)
        # self.conv1 = nn.Conv2d(256, 256, kernel_size=3,padding=1, stride=1)
        # self.BN1 = nn.BatchNorm2d(256, eps=1e-5, affine=True)
        # self.dropout1 = nn.Dropout(0.3)
        # self.conv2 = nn.Conv2d(128,64, kernel_size=3,padding=1, stride=1)
        # self.BN2 = nn.BatchNorm2d(64, eps=1e-5, affine=True)
        # self.dropout2 = nn.Dropout(0.3)
        # self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        # self.BN3 = nn.BatchNorm2d(32, eps=1e-5, affine=True)
        # self.dropout3 = nn.Dropout(0.3)
        # self.conv4 = nn.Conv2d(32, 16, kernel_size=3,padding=1, stride=1)
        # self.BN4 = nn.BatchNorm2d(16, eps=1e-5, affine=True)
        # self.dropout4 = nn.Dropout(0.3)
        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1)
        self.BN5 = nn.BatchNorm2d(8, eps=1e-5, affine=True)
        self.dropout5 = nn.Dropout(0.3)
        self.conv6 = nn.Conv2d(8, 4, kernel_size=3, padding=1, stride=1)
        self.BN6 = nn.BatchNorm2d(4, eps=1e-5, affine=True)
        self.dropout6 = nn.Dropout(0.3)
        self.conv7 = nn.Conv2d(4, 2, kernel_size=3, padding=1, stride=1)
        self.BN7 = nn.BatchNorm2d(2, eps=1e-5, affine=True)
        self.dropout7 = nn.Dropout(0.3)
        # 定义 FusionModel
        self.fc = nn.Linear(4, 8)
        self.fc1 = nn.Linear(64, 2)
        self.fc2 = nn.Linear(25, 2)
        # GLU output layer
        self.linear1 = nn.Linear(4, 2)
        self.linear2 = nn.Linear(2, 4)
        self.sigmoid = nn.Sigmoid()
        # 线性注意力融合
        self.fusion = EnhancedCrossAttention(d_model=64, nhead=8, dropout=0.1)
        # # 添加全局残差投影层
        # self.residual_proj = nn.Linear(2, 2)
    # def prepare_gatv2_input(self,nodes_abs, patch_len:int, stride, dst=None):
    #     """
    #     准备 GATv2Conv 的输入数据。 # X 节点特征，形状 (num_Ped, 2)
    #     :param nodes_abs: 归一化节点坐标，形状 (seq_length, num_Ped, 2)
    #     :param nei_list: 邻居列表，形状 (seq_length, num_Ped, max_nei_num)
    #     :param batch_pednum: 每个场景的行人数量，形状 (batch_size)
    #     :param seq_list: 时间步t = nodes_abs.shape[0]
    #     :return: x, edge_index, edge_attr
    #     """
    #     num_Ped = nodes_abs.shape[1]

    #     # # 构建边索引
    #     # edge_index = torch.nonzero(nei_list)
    #     # edge_index = edge_index.clone().detach().t().contiguous()
    #     # edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     # 转换为 (2, num_edges)it is recommended to use sourceTensor.clone().detach()

    #     decompose = nodes_abs.unfold(dimension=1, size=patch_len, step=stride)
    #     decompose = rearrange(decompose, 'n l d m -> l n m d')
    #     edge_list = []
    #     data_list = []
    #     edge_index_for_edge = []

    #     for i in range(decompose.shape[0]):
    #         coordinates = decompose[i]
    #         coordinates = rearrange(coordinates, 'n t d -> (n t) d')
    #         dist_matrix = torch.cdist(coordinates, coordinates, p=2)

    #         epsilon = 1e-6
    #         dist_matrix += epsilon

    #         adj_matrix = torch.zeros_like(dist_matrix)

    #         adj_matrix = (dist_matrix < dst).float()
    #         adj_matrix.fill_diagonal_(0)

    #         edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()
    #         edge_attr = dist_matrix[edge_index[0], edge_index[1]]
    #         edge_index, edge_attr = remove_self_loops(edge_index, edge_attr=edge_attr)

    #         start_nodes = edge_index[0].unsqueeze(1).expand(-1, edge_index.size(1)).to(edge_index.device)
    #         end_nodes = edge_index[1].unsqueeze(1).expand(-1, edge_index.size(1)).to(edge_index.device)
    #         shared_start = start_nodes == start_nodes.t()
    #         shared_end = end_nodes == end_nodes.t()
    #         shared_mixed = (start_nodes == end_nodes.t()) | (end_nodes == start_nodes.t()).to(edge_index.device)

    #         shared_nodes_mask = (shared_start | shared_end | shared_mixed) & ~torch.eye(edge_index.size(1),
    #                                                                                     dtype=torch.bool).to(
    #             edge_index.device)

    #         edge_graph_indices = shared_nodes_mask.nonzero(as_tuple=False)
    #         edge_graph_index = edge_graph_indices.t().contiguous()

    #         edge_list.append(edge_index)
    #         data_list.append(edge_attr)
    #         edge_index_for_edge.append(edge_graph_index)
    #         #edge_index, edge_attr, edge_edge_index  = pdd_matrix_edges(absolute_trajectory, patch_len, stride, 2)

    #     return edge_list, data_list, edge_index_for_edge

    def forward(self, inputs, iftest=False):
        # DYNA output
        dyna_output = self.dyna(inputs, iftest)  # unsq[T, N, 2]>[1,T,N,2] 4
        # dyna_output_0 = self.dyna.forward(inputs, iftest)[0].unsqueeze(0)  # [T, N, 2]>[1,T,N,2]增加了dyna的输出nodeabs,neilist
        # dyna_output_1 = self.residual_0(dyna_output_0.permute(0,3,2,1))
        # dyna_output = dyna_output.permute(2, 1, 0) # [2, N, T]
        # dyna_output = dyna_output.unsqueeze(0) # [1,2, N, T]
        # dyna_output =  self.dropout0(F.relu(self.BN0(self.conv0(dyna_output))))
        # dyna_output = dyna_output.squeeze(0)
        # dyna_output = dyna_output.permute(2, 1, 0) # [2, N, T]
        # #dyna_output = self.output_layer(dyna_output_2)
        # dyna_output = dyna_output_2.squeeze(0).permute(2,1,0)

        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs

        # GAT output
        spatial_features = nodes_norm[-1]  # Use the last frame's normalized coordinates
        adj = nei_lists[-1]  # Use the last frame's adjacency matrix
        # Convert adj to edge_index format
        edge_index = torch.nonzero(adj).t().long()  # [2, E]转置 # Convert to integer type
        edge_index = edge_index.clone().detach().contiguous()

        # 25/3/5 Change the RES directly from 256 to 2 through conv
        gat_output = self.gat(spatial_features, edge_index)  # [N, 64]
        gat_output = gat_output.unsqueeze(0).repeat(nodes_norm.shape[0], 1, 1)  # [T, N, 64]
        # gat_output = gat_output.permute(2, 1, 0) # [64, N, T]
        # gat_output = gat_output.unsqueeze(0) # [1,64, N, T]
        # # gat_output = self.dropout2(F.relu(self.BN2(self.conv2(gat_output))))
        # gat_output = self.dropout5(F.relu(self.BN5(self.conv5(gat_output))))
        # gat_output = self.dropout6(F.relu(self.BN6(self.conv6(gat_output))))
        # gat_output = self.dropout7(F.relu(self.BN7(self.conv7(gat_output))))
        # gat_output = gat_output.squeeze(0)
        # gat_output = gat_output.permute(2, 1, 0) # [T, N, 2]
        gat_output = F.relu(self.fc1(gat_output))
        # gat_output =F.relu(self.fc2(gat_output))

        # gat_output = gat_output.unsqueeze(0).repeat(nodes_norm.shape[0], 1, 1)  # [T, N, 256]
        # gat_output = self.residual(gat_output.permute(0,3,2,1))#[1,T,N,256]>[1,256,N,T]
        '''
        gat_output_0 = gat_output.permute(1, 0) # [256, N]
        gat_output_1 = gat_output_0.unsqueeze(-1).repeat(1, 1,nodes_abs.shape[0] )  # [256, N,T]
        gat_output_2 = gat_output_1.unsqueeze(0) # [1,256, N,T]
        gat_output = self.residual(gat_output_2)  # [1,256,N,T]
        gat_output = F.relu(self.BN1(self.conv1(gat_output)))
        gat_output = F.relu(self.BN2(self.conv2(gat_output)))
        gat_output = F.relu(self.BN3(self.conv3(gat_output)))
        gat_output = F.relu(self.BN4(self.conv4(gat_output)))
        gat_output = F.relu(self.BN5(self.conv5(gat_output)))
        '''
        # gat_output = self.output_layer(gat_output)

        # # Residual connection
        # output = dyna_output + self.output_layer(gat_output)  # [T, N, 2]
        # output =torch.cat([dyna_output, gat_output.squeeze(0).permute(2, 1, 0)], dim=1)only match in dim=-1,so no
        # output = torch.cat([dyna_output, gat_output.squeeze(0).permute(2, 1, 0)], dim=1)Sizes of tensors must match except in dimension 1.
        # Expected size 19 but got size 20 for tensor number 1 in the list
        # gat_output = gat_output.squeeze(0).permute(2, 1, 0)#[2,264,20] [2,N+1,T+1]>[20,264,2]
        # gat_output_final = gat_output[:edge_index.shape, :batch_norm.shape[0], :]

        # ----------------------------------------------------------------------------------------------------------------------
        # # 3的输出
        # combined = torch.cat((dyna_output, gat_output), dim=-1)  # 形状 (T, N, 4)
        # hidden = torch.relu(self.fc1(combined))  # 形状 (T, N, hidden_dim)
        # output = self.fc2(hidden)  # 形状 (T, N, 2)

        #  # 6 GLU:25/3/11
        # combined = torch.cat((dyna_output, gat_output), dim=-1)  # 形状 (T, N, 4)
        # gate = self.sigmoid(self.linear2(combined)) # 门控信号
        # output = self.linear1(combined)  # 线性变换

        # 测试线性注意力的结果20/5/19
        output = self.fusion(dyna_output, gat_output)
        # # 保存原始输入用于残差连接
        # output = output + self.residual_proj(spatial_features)
        # # output = torch.stack([dyna_output, gat_output], dim=2).sum(dim=2)  # 形状: (T, N, 2)

        # output = self.BNAftercat(self.Aftercat(output.permute(0,3,1,2)))针对cat维度变化的时候反卷积

        return output

