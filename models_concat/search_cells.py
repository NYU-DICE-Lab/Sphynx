""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models_concat import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, config, n_nodes, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.duplicate = False
        self.n_nodes = n_nodes
        self.config = config

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        # if reduction_p:
        #     self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        # else:
        #     self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        # self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)
        
        # if reduction_p:
        #     if config.pre_relu:
        #         self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        #     else:
        #         self.preproc0 = ops.FactorizedReduce_noReLU(C_pp, C, affine=False)
        # else:
        #     if config.pre_relu:
        #         self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        #     else:
        #         self.preproc0 = ops.StdConv_noReLU(C_pp, C, 1, 1, 0, affine=False)
        # if config.pre_relu:
        #     self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)
        # else:
        #     self.preproc1 = ops.StdConv_noReLU(C_p, C, 1, 1, 0, affine=False)

        if reduction_p:
            self.preproc0 = ops.FactorizedReduce_noReLU(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv_noReLU(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv_noReLU(C_p, C, 1, 1, 0, affine=False)
        
        self.proproc = ops.StdConv(4*C, C, 1, 1, 0, affine=False)
        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i): # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                # if self.config.ops_relu:
                    # op = ops.MixedOp(C, stride, config)
                # elif self.config.reduce_norelu and self.reduction:
                    # op = ops.MixedOp_noReLU(C, stride, config)
                # else:
                op = ops.MixedOp_noReLU(C, stride, config)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)
        s_out = self.proproc(s_out)

        return s_out

class SearchCell_noReLU(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, config, n_nodes, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.duplicate = True
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce_noReLU(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv_noReLU(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv_noReLU(C_p, C, 1, 1, 0, affine=False)

        self.proproc = ops.StdConv_noReLU(4*C, C, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i): # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = ops.MixedOp_noReLU(C, stride, config)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)
        s_out = self.proproc(s_out)
        return s_out