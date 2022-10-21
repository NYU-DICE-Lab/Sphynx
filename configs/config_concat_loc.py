""" Config class for search/augment """
import argparse
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import genotypes.genotypes_concat as gt
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=96, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)                        # default: 16 channels
        parser.add_argument('--layers', type=int, default=8, help='# of layers')            # default: 8 layers
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--beta_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--beta_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')
        # parser.add_argument('--early_stopping', type=int, default=100)
        parser.add_argument('--tau_max', type=float, default=10.0)
        parser.add_argument('--tau_min', type=float, default=0.1)
        parser.add_argument('--genotype', help='Cell genotype', required=True)
        parser.add_argument('--print_relu', action='store_true', help='Print output shape after relu layers')
        parser.add_argument('--subsampling', type=int, default=3)
        # parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        # parser.add_argument('--ops_relu', action='store_true', help='ReLU in the OPS')
        # parser.add_argument('--pre_relu', action='store_true', help='ReLU in the preprocessing: 1x1 convolution')
        # parser.add_argument('--reduce_norelu', action='store_true')
        # parser.add_argument('--reduce_loc', help='reduce cell location', default="[2, 4]")
        # parser.add_argument('--no_reduce', action='store_true', help='Replace with residual block')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('searchs', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.genotype = gt.from_str(self.genotype)
        self.gpus = parse_gpus(self.gpus)


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=96, help='batch size')
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')    # default: 600
        parser.add_argument('--init_channels', type=int, default=36)            # default: 36
        parser.add_argument('--layers', type=int, default=20, help='# of layers')    # default: 20
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=0, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
        # PC-DARTS Version
        parser.add_argument('--genotype', help='Cell genotype', default="Genotype(normal=[[('van_conv_3x3', 0), ('van_conv_3x3', 1)], [('van_conv_3x3', 0), ('van_conv_3x3', 1)], [('skip_connect', 0), ('van_conv_3x3', 1)], [('skip_connect', 0), ('dil_conv_3x3', 2)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('skip_connect', 2), ('max_pool_3x3', 1)], [('max_pool_3x3', 1), ('skip_connect', 2)], [('max_pool_3x3', 1), ('skip_connect', 2)]], reduce_concat=range(2, 6))")
        parser.add_argument('--cryptonas_space', action='store_true', help='Activating the CryptoNAS rule')
        parser.add_argument('--print_relu', action='store_true', help='Print output shape after relu layers')
        # parser.add_argument('--relu_prune', action='store_true', help='Prune the relu on preprocessing')
        parser.add_argument('--relu_ops', help='relu operations', default="['van_conv_3x3', 'van_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']")
        parser.add_argument('--reduce_loc', help='reduce cell location', default="[2, 4]")
        parser.add_argument('--pre_relu', action='store_true', help="ReLU in the preprocessing")
        parser.add_argument('--reduce_norelu', action='store_true', help="No ReLU iin the reduce block")
        return parser

        """ 
        Genotype(normal=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], [('dil_conv_5x5', 2), ('skip_connect', 0)], [('skip_connect', 0), ('dil_conv_5x5', 1)], [('dil_conv_3x3', 0), ('dil_conv_5x5', 1)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('skip_connect', 2), ('max_pool_3x3', 1)], [('skip_connect', 3), ('skip_connect', 2)], [('skip_connect', 2), ('avg_pool_3x3', 1)]], reduce_concat=range(2, 6))"
        PC-DARTS: Genotype(normal=[[('skip_connect', 0), ('van_conv_3x3', 1)], [('van_conv_3x3', 0), ('dil_conv_3x3', 1)], [('van_conv_5x5', 0), ('van_conv_3x3', 1)], [('avg_pool_3x3', 0), ('dil_conv_3x3', 1)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('van_conv_5x5', 1)], [('van_conv_5x5', 1), ('skip_connect', 2)], [('van_conv_3x3', 0), ('van_conv_3x3', 2)], [('van_conv_3x3', 1), ('van_conv_3x3', 2)]], reduce_concat=range(2, 6))
        Genotype(normal=[[('skip_connect', 0), ('sep_conv_3x3_CNS', 1)], [('sep_conv_3x3_CNS', 0), ('dil_conv_3x3_NR', 1)], [('sep_conv_5x5_CNS', 0), ('sep_conv_3x3_CNS', 1)], [('avg_pool_3x3', 0), ('dil_conv_3x3_NR', 1)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5_CNS', 1)], [('sep_conv_5x5_CNS', 1), ('skip_connect', 2)], [('sep_conv_3x3_CNS', 0), ('sep_conv_3x3_CNS', 2)], [('sep_conv_3x3_CNS', 1), ('sep_conv_3x3_CNS', 2)]], reduce_concat=range(2, 6))
        Genotype(normal=[[('skip_connect', 0), ('sep_conv_3x3_NR', 1)], [('sep_conv_3x3_NR', 0), ('dil_conv_3x3_NR', 1)], [('sep_conv_5x5_NR', 0), ('sep_conv_3x3_NR', 1)], [('avg_pool_3x3', 0), ('dil_conv_3x3_NR', 1)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5_NR', 1)], [('sep_conv_5x5_NR', 1), ('skip_connect', 2)], [('sep_conv_3x3_NR', 0), ('sep_conv_3x3_NR', 2)], [('sep_conv_3x3_NR', 1), ('sep_conv_3x3_NR', 2)]], reduce_concat=range(2, 6))
        
        All SKIP CONNECT:
        Genotype(normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 1), ('skip_connect', 2)], [('skip_connect', 1), ('skip_connect', 2)], [('skip_connect', 1), ('skip_connect', 2)]], normal_concat=range(2, 6), reduce=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 1), ('skip_connect', 2)], [('skip_connect', 1), ('skip_connect', 2)], [('skip_connect', 1), ('skip_connect', 2)]], reduce_concat=range(2, 6))        
        """
        
    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('augments', self.name)
        self.genotype = gt.from_str(self.genotype)
        self.gpus = parse_gpus(self.gpus)