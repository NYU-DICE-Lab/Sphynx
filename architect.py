""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
from torch.nn import functional as F


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def relu_penalty(self, arch_params):
        num_nodes = 4
        relu_vector = torch.tensor([0., 0., 0., 1., 1., 1., 1., 0.]).cuda(non_blocking=True)
        relu_budget = 0
        for i in range(num_nodes+1):
            sub_budget = 0
            for j in range(4):
                try:
                    sub_budget += torch.mm(F.softmax(arch_params[j][i], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                except:
                    continue

            relu_budget += sub_budget/((5-i)**2)

        return relu_budget


    # It would be good if the relu_budget applies the relu-shuffling
    # This needs to be fixed. 
    def relu_budget(self, arch_params):
        num_nodes = 4
        relu_vector = torch.tensor([0., 0., 0., 1., 1., 1., 1., 0.]).cuda(non_blocking=True)
        relu_budget = 0
        # print(arch_params)
        
        # for arch in arch_params:
        #     print(arch)
        #     break

        for i in range(num_nodes):
            if i == 0:
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][0], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][1], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
            elif i == 1:
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][0], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][1], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][2], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
            elif i == 2:
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][0], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][1], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][2], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][3], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
            elif i == 3:
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][0], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][1], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][2], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][3], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
                relu_budget += torch.mm(torch.nn.functional.softmax(arch_params[i][4], dim=0).unsqueeze(0), relu_vector.unsqueeze(1))
            else:
                raise ValueError('In the relu_budget, the index cannot be bigger than 3')

        return relu_budget


    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, config):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        # if config.relu_coefficient <= 1e-8:
        loss = self.v_net.loss(val_X, val_y) # L_val(w`)
        # else:
        #     loss = self.v_net.loss(val_X, val_y)
        #     # arch_normal = []
        #     # for alpha in self.v_net.alpha_normal:
        #         # arch_normal.append(alpha)
        #     # relu_loss = self.relu_budget(arch_normal)
        #     relu_loss = 0.
        #     relu_loss += self.relu_penalty(self.v_net.alpha_normal)
        #     relu_loss += self.relu_penalty(self.v_net.alpha_reduce)
        #     print("ReLU Loss:", relu_loss)
        #     loss = loss + config.relu_coefficient * relu_loss

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y, config)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, trn_X, trn_y, config):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        # if config.cryptonas_space == False:
        loss = self.net.loss(trn_X, trn_y)
        # # else:
        #     loss = self.net.loss(trn_X, trn_y)
        #     arch_normal = []
        #     for alpha in self.v_net.alpha_normal:
        #         arch_normal.append(alpha)
        #     relu_loss = self.relu_budget(arch_normal)
        #     loss = loss + config.relu_coefficient * relu_loss

        # loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian