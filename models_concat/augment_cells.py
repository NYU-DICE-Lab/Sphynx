""" CNN cell for network augmentation 
    This file cooresponds to augment_cnn2.py and augment3.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models_concat import ops
import genotypes.genotypes_concat as gt


# THIS NOT USED. IGNORE.
class AugmentCell_Preproc0(nn.Module):

    def __init__(self, C_pp, C):
        super().__init__()
        # self.reduction = reduction
        # self.reduction_p = reduction_p

        # self.config = config

        # if reduction_p:
        self.preproc0 = ops.FactorizedReduce_noReLU(C_pp, C, affine=False)
        # else:
            # self.preproc0 = ops.StdConv_noReLU(C_pp, C, 1, 1, 0, affine=False)
        # self.preproc1 = ops.StdConv_noReLU(C_p, C, 1, 1, 0, affine=False)

    def forward(self, s0):
        s0 = self.preproc0(s0)
        # s1 = self.preproc1(s1)
        return s0

# THIS NOT USED. IGNORE.
class AugmentCell_Preproc1(nn.Module):

    def __init__(self, C_p, C):
        super().__init__()

        self.preproc1 = ops.StdConv_noReLU(C_p, C, 1, 1, 0, affine=False)

    def forward(self, s1):
        s1 = self.preproc1(s1)
        return s1

# THIS NOT USED. IGNORE.
class AugmentCell_NoPrep(nn.Module):
    """
    This won't have the preprocessing layers here. 
    """

    def __init__(self, genotype, C, reduction):
        super().__init__()
        self.n_nodes = len(genotype.normal)

        self.proproc = ops.StdConv_noReLU(4*C, C, 1, 1, 0, affine=False)

        if reduction: 
            gene = genotype.reduce
            self.concat = genotype.reduce_concat

        else:
            gene = genotype.normal
            self.concat = genotype.normal_concat

        self.genotype = gene 

        self.dag = gt.to_dag(C, gene, reduction)

    def forward(self, s0, s1):
        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        s_out = self.proproc(s_out)

        s_out = F.relu(s_out)
        return s_out
        

# USED IN augment_cnn.py
class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """
    def __init__(self, config, genotype, C_pp, C_p, C, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.reduction_p = reduction_p
        self.n_nodes = len(genotype.normal)
        # print("number of nodes: ", self.n_nodes)
        self.config = config

        if reduction_p:
            self.preproc0 = ops.FactorizedReduce_noReLU(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv_noReLU(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv_noReLU(C_p, C, 1, 1, 0, affine=False)

        self.proproc = ops.StdConv_noReLU(4*C, C, 1, 1, 0, affine=False)


        # if self.config.pre_relu:
        #     self.proproc = ops.StdConv(4*C, C, 1, 1, 0, affine=False)
        # else:
        #     self.proproc = ops.StdConv_noReLU(4*C, C, 1, 1, 0, affine=False)

        # generate dag
        if reduction:
            gene = genotype.reduce
            # gene = genotype.normal
            self.concat = genotype.reduce_concat
        else:
            gene = genotype.normal
            self.concat = genotype.normal_concat
        self.genotype = gene
        
        # if self.config.reduce_norelu and self.reduction:
            # self.dag = gt.to_dag_drop_relu(C, gene, self.reduction)
        # else:
        self.dag = gt.to_dag(C, gene, self.reduction)

    def forward(self, s0, s1):

        # print("In the AugmentCell") 
        # print("self.reduction:", self.reduction)
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        # print("1")
        states = [s0, s1]
        # print("2")
        # for edges in self.dag:
            # print(edges)
        # print("2")
        # if self.reduction:
            # print(self.dag)
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)
        # print("3")
        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        s_out = self.proproc(s_out)

        # if self.reduction == False:
        # if self.config.reduce_norelu == True and self.reduction == True:
        #     return s_out
        # else:
        s_out = F.relu(s_out)
        if self.config.print_relu:
            # print("---------------")
            print(s_out.shape[1]*s_out.shape[2]*s_out.shape[3])
                # print("---------------")
        return s_out


        # states_relu = []

        # if self.config.reduce_norelu:
        #     if self.reduction:
        #         relu_ops = []
        #     else:
        #         relu_ops = eval(self.config.relu_ops)
        # else:
        #     relu_ops = eval(self.config.relu_ops)

        # # relu_ops = eval(self.config.relu_ops)

        # states_flag = []
        # # adding the relu checking 
        # for i in range(self.n_nodes+1):    # Total node - output node - last intermediate node
        #     relu_flag = False
        #     for nodes in self.genotype:
        #         for op_name in nodes: 
        #             if op_name[0] in relu_ops and op_name[1] == i:
        #                 relu_flag = True
        #                 break
        #         if relu_flag == True:
        #             break
        #     states_flag.append(relu_flag)
        
        # states_flag.append(False)    # Last intermediate node
        # # print(states_flag)
        # # print("states_flag: ", states_flag)
        # # first working on the first two input nodes:
        # for i in range(2):
        #     if states_flag[i] == True:
        #         after_relu = F.relu(states[i])
        #         states_relu.append(after_relu)

        #         if self.config.print_relu == True:
        #             print(after_relu.shape[1]*after_relu.shape[2]*after_relu.shape[3])
        #     else:
        #         states_relu.append(None)

        # # now computing the intermediate nodes: 
        # for i, edges in enumerate(self.dag):
        #     sum_list = []
        #     for j, op in enumerate(edges):
        #         if self.genotype[i][j][0] not in relu_ops:
        #             sum_list.append(op(states[op.s_idx]))
        #         else:
        #             sum_list.append(op(states_relu[op.s_idx]))
                   
        #     states.append(sum(sum_list))
        #     if states_flag[i+2] == True:
        #         after_relu = F.relu(states[i+2])
        #         states_relu.append(F.relu(states[i+2]))
        #         # TODO: add print relu
        #         if self.config.print_relu == True:
        #             print(after_relu.shape[1]*after_relu.shape[2]*after_relu.shape[3])
        #     else:
        #         states_relu.append(None)
            
        # s_out = torch.cat([states[i] for i in self.concat], dim=1)
        # s_out = self.proproc(s_out)
        # if self.reduction:
        #     return s_out
        # else:
        #     s_out = F.relu(s_out)
        #     if self.config.print_relu:
        #         print(s_out.shape[1]*s_out.shape[2]*s_out.shape[3])
        #         # print("Hello")

        # # if self.config.print_relu and self.config.pre_relu and self.reduction == False:
        #     # print(s_out.shape[1]*s_out.shape[2]*s_out.shape[3])
        # return s_out

class AugmentCell_NoReLU(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """
    def __init__(self, config, genotype, C_pp, C_p, C, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.duplicate = True
        self.n_nodes = len(genotype.normal)
        self.config = config

        # print("C_pp, C_p, C: ", C_pp, C_p, C)

        # TODO: Replace with the operations which matches the size. 

        if reduction_p:
            self.preproc0 = ops.FactorizedReduce_noReLU(C_pp, C)
        else:
            self.preproc0 = ops.StdConv_noReLU(C_pp, C, 1, 1, 0)
        self.preproc1 = ops.StdConv_noReLU(C_p, C, 1, 1, 0)

        self.proproc = ops.StdConv_noReLU(4*C, C, 1, 1, 0, affine=False)
        
        gene = genotype.double
        self.concat = genotype.double_concat

        self.genotype = gene
        self.dag = gt.to_dag(C, gene, reduction)

    def forward(self, s0, s1):
        # print("In the AugmentCell_norelu") 
        # print("self.reduction:", self.reduction)

        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        s_out = self.proproc(s_out)
        return s_out

        # states_relu = []

        # # relu_ops = eval(self.config.relu_ops)
        # relu_ops = []
        # states_flag = []
        # # adding the relu checking 
        # for i in range(5):    # Total node - output node - last intermediate node
        #     relu_flag = False
        #     for nodes in self.genotype:
        #         for op_name in nodes: 
        #             if op_name[0] in relu_ops and op_name[1] == i:
        #                 relu_flag = True
        #                 break
        #         if relu_flag == True:
        #             break
        #     states_flag.append(relu_flag)
        
        # states_flag.append(False)    # Last intermediate node

        # ###################################################################################
        
        # # else:
        # for i in range(2):
        #     if states_flag[i] == True:
        #         after_relu = F.relu(states[i])
        #         states_relu.append(after_relu)
        #         # TODO: add print relu
        #         if self.config.print_relu == True:
        #             print(after_relu.shape[1]*after_relu.shape[2]*after_relu.shape[3])
        #     else:
        #         states_relu.append(None)

        # # now computing the intermediate nodes: 
        # for i, edges in enumerate(self.dag):
        #     sum_list = []
        #     for j, op in enumerate(edges):
        #         if self.genotype[i][j][0] not in relu_ops:
        #             sum_list.append(op(states[op.s_idx]))
        #         else:
        #             sum_list.append(op(states_relu[op.s_idx]))

        #     states.append(sum(sum_list))
        #     if states_flag[i+2] == True:
        #         after_relu = F.relu(states[i+2])
        #         states_relu.append(F.relu(states[i+2]))
        #         # TODO: add print relu
        #         if self.config.print_relu == True:
        #             print(after_relu.shape[1]*after_relu.shape[2]*after_relu.shape[3])
        #     else:
        #         states_relu.append(None)

        # # print(states_relu)
        # ###################################################################################
        
        # s_out = torch.cat([states[i] for i in self.concat], dim=1)
        # s_out = self.proproc(s_out)

        # return s_out
