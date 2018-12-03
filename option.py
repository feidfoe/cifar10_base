# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('-e', '--exp_name',   required=True,              help='experiment name')

## Arch
parser.add_argument('--n_class',          default=10,     type=int,   help='number of classes')
parser.add_argument('--input_size',       default=32,     type=int,   help='input size')
parser.add_argument('--batch_size',       default=128,    type=int,   help='mini-batch size')
parser.add_argument('--arch',             default='resnet',           help='architecture to use')
parser.add_argument('--depth',            default=18,     type=int,   help='depth of network')


## Logging
parser.add_argument('--checkpoint',       default=None,               help='checkpoint to resume')
parser.add_argument('--log_step',         default=50,     type=int,   help='step for logging in iteration')
parser.add_argument('--save_step',        default=50,     type=int,   help='step for saving in epoch')
parser.add_argument('--save_dir',         default='./checkpoint/',    help='save directory for checkpoint')


## Data
parser.add_argument('--data_dir',         default='./dataset/',       help='data directory')
parser.add_argument('--data_split',       default='train',            help='data split to use')


## Optimization
parser.add_argument('--lr',               default=0.1,   type=float, help='initial learning rate')
parser.add_argument('--lr_decay_rate',    default=0.1,    type=float, help='lr decay rate')
parser.add_argument('--lr_decay_period',  default=150,    type=int,   help='lr decay period')
parser.add_argument('--momentum',         default=0.9,    type=float, help='sgd momentum')
parser.add_argument('--weight_decay',     default=0.0005, type=float, help='sgd optimizer weight decay')
parser.add_argument('--max_step',         default=350,    type=int,   help='maximum step for training')
parser.add_argument('--is_train',         action='store_true',        help='whether it is training')


## Flag
parser.add_argument('--cuda',             action='store_true',        help='enables cuda')
parser.add_argument('-d', '--debug',      action='store_true',        help='debug mode')
parser.add_argument('--use_pretrain',     action='store_true',        help='whether it use pre-trained parameters if exists')
parser.add_argument('--random_seed',                      type=int,   help='random seed')
parser.add_argument('--num_workers',      default=4,      type=int,   help='number of workers in data loader')
parser.add_argument('--cudnn_benchmark',  default=True,   type=bool,  help='cuDNN benchmark')
parser.add_argument('-g', '--gpu_ids',    default=[0],    type=lambda x: [int(i) for i in x.split(',')], 
                                                                      help='GPUs to use')























def get_option():
    option = parser.parse_args()
    return option
