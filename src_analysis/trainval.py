import argparse
import ast
import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))

print(sys.path)

import torch
import yaml
import random
from src.processor import processor
from src.goal_estimator import *

# Use Deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='DDL')
    parser.add_argument('--dataset', default='eth5')
    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--config')
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    parser.add_argument('--test_set', default='eth', type=str,
                        help='Set this value to [eth, hotel, zara1, zara2, univ] for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument('--base_dir', default='.', help='Base directory including these scripts.')
    parser.add_argument('--save_base_dir', default='../outputs/', help='Directory for saving caches and models.')
    parser.add_argument('--phase', default='test', help='Set this value to \'train\' or \'test\'')
    parser.add_argument('--train_model', default='ddl', help='Your model name')
    parser.add_argument('--load_model', default=58, type=str, help="load pretrained model for test or training")
    parser.add_argument('--model', default='star.STAR')
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=8, type=int)
    parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--batch_around_ped', default=256, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--show_step', default=1, type=int)
    parser.add_argument('--start_test', default=0, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval)
    parser.add_argument('--ifsave_results', default=False, type=ast.literal_eval)
    parser.add_argument('--randomRotate', default=True, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--neighbor_thred', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.0015, type=float)
    parser.add_argument('--clip', default=1, type=int)
    parser.add_argument('--only_show_result', default=False, type=ast.literal_eval, help="=True:there is analysis data that is stored")
    parser.add_argument('--calculate_gradient', default=False, type=ast.literal_eval, help="=True:calculate the time gradient")



    return parser


def load_arg(p):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    p.save_dir = p.save_base_dir + str(p.test_set) + '/'
    p.model_dir = p.save_base_dir + str(p.test_set) + '/' + p.train_model + '/'
    p.config = p.model_dir + '/config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)

    args = load_arg(p)

    torch.cuda.set_device(0)

    trainer = processor(p)


    save_directory = "./goal_estimated_results/eth_ucy/"

    DATASET_NAME_TO_NUM = {
        'eth': 0,
        'hotel': 1,
        'zara1': 2,
        'zara2': 3,
        'univ': 4
    }
    test_set_name = list(DATASET_NAME_TO_NUM.keys())[p.test_set]
    file_path = os.path.join(save_directory, f'goal_estimated_{test_set_name}.pkl')
    if not (os.path.exists(file_path)):
        print('the path to predicted endpoint is wrong')
    else:
        trainer.test()



