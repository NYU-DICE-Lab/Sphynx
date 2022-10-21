""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from collections import namedtuple
import torch
import torch.nn as nn
# from models_cns import ops
from models_concat import ops
from glob import glob

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    # 'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect', # identity
    'van_conv_3x3',
    'van_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none'
]

# PRIMITIVES_noReLU = [
#     # 'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect_NR', # identity
#     'van_conv_3x3_NR',
#     'van_conv_5x5_NR',
#     'dil_conv_3x3_NR',
#     'dil_conv_5x5_NR',
#     'none'
# ]

# PRIMITIVES_noReLU = [
#     # 'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect_NR', # identity
#     'sep_conv_3x3_NR',
#     'sep_conv_5x5_NR',
#     'dil_conv_3x3_NR',
#     'dil_conv_5x5_NR',
#     'none'
# ]

PRIMITIVES_noReLU = [
    # 'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect_NR', # identity
    'van_conv_3x3_NR',
    'van_conv_5x5_NR',
    'van_dil_conv_3x3_NR',
    'van_dil_conv_5x5_NR',
    'none'
]


def to_dag(C_in, gene, reduction):
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True)
            if not isinstance(op, ops.Identity):
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)
    return dag

def to_dag_drop_relu(C_in, gene, reduction):
    """
    Convert the original operations in PRIMITIVES with NR.
    The purpose is that there is the relu shuffling going so we need to remove the RELU in the operations.
    """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            stride = 2 if reduction and s_idx < 2 else 1
            if op_name in ['van_conv_3x3', 'van_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']:
                op = ops.OPS[op_name+'_NR'](C_in, stride, True)
            else:
                op = ops.OPS[op_name](C_in, stride, True)

            if not isinstance(op, ops.Identity):
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)
    return dag

def from_str(s):
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """

    genotype = eval(s)

    return genotype


def parse(config, alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'
    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            # if config.cryptonas_space == True:
                # prim = /PRIMITIVES_CRYPTONAS[prim_idx]
            prim = PRIMITIVES[prim_idx]
            # else:
                # prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))
        gene.append(node_gene)
    return gene

def parse_noReLU(config, alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES_noReLU[-1] == 'none' # assume last PRIMITIVE is 'none'
    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            # if config.cryptonas_space == True:
                # prim = /PRIMITIVES_CRYPTONAS[prim_idx]
            prim = PRIMITIVES_noReLU[prim_idx]
            if prim == 'skip_connect':
                prim = prim + '_NR'
            # else:
                # prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))
        gene.append(node_gene)
    return gene