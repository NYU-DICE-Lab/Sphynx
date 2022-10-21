""" CNN for network augmentation """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_concat.augment_cells import AugmentCell, AugmentCell_NoReLU
from models_concat import ops


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.AvgPool2d(5, stride=input_size-5, padding=0, count_include_pad=False), # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 768, kernel_size=2, bias=False), # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=False)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class AuxiliaryHeadTinyImageNet(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """Assuming input size 16x16"""
        assert input_size in [16]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.AvgPool2d(5, stride=6, padding=0, count_include_pad=False), # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 768, kernel_size=2, bias=False), # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=False)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, input_size, C, n_classes):
        assert input_size in [14]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.ReLU()
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)    # Flatten
        logits = self.linear(out)
        return logits




class AugmentCNNImageNet(nn.Module):
    """Augment CNN model for ImageNet"""

    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype, config, 
                 stem_multiplier=3):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.config = config
        # aux head position
        self.aux_pos = 4*n_layers//6 if auxiliary else -1

        if self.config.imagenet_stem_relu:
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C // 2),
                nn.ReLU(),
                nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C)
            )

            self.stem1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C)
            )
        else:
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C // 2),
                nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C)
            )

            self.stem1 = nn.Sequential(
                nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C)
            )
        
        C_pp, C_p, C_cur = C, C, C
        
        # Saving this for the forward pass.
        self.C_pp = C_pp
        self.C_p = C_p
        self.C_cur = C_cur

        self.cells = nn.ModuleList()
        reduction_p = True
        reduce_location = eval(config.reduce_loc)

        if reduce_location[1] > 2*n_layers//3:
            raise ValueError("second index of reduce location should be small or equal than 2*n_layers//3")

        for i in range(n_layers):
            # if i in [reduce_location[0]*n_layers//6, reduce_location[1]*n_layers//6]:
            if i in reduce_location:
                C_cur *= 4
                reduction = True
            else: 
                reduction = False

            cell = AugmentCell(config, genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur
            C_pp, C_p = C_p, C_cur_out

            if i == self.aux_pos:
                if config.dataset == 'tiny_imagenet':
                    self.aux_head = AuxiliaryHeadTinyImageNet(input_size // 4, C_p, n_classes)
                elif config.dataset == 'imagenet':
                    self.aux_head = AuxiliaryHeadImageNet(input_size // 16, C_p, n_classes)
                else:
                    self.aux_head = AuxiliaryHead(input_size//4, C_p, n_classes)
        self.gap = nn.AvgPool2d(7)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x):
        # s0 = s1 = self.stem(x)
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        if self.config.imagenet_stem_relu and self.config.print_relu:
            print((self.C // 2) *112*112)
            print(self.C*56*56)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)
                
        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits, aux_logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p



class AugmentCNN(nn.Module):
    """ Augmented CNN model """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype, config, 
                 stem_multiplier=3):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        self.config = config
        # aux head position
        self.aux_pos = 4*n_layers//6 if auxiliary else -1

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )
        
        C_pp, C_p, C_cur = C_cur, C_cur, C
        
        # Saving this for the forward pass.
        self.C_pp = C_pp
        self.C_p = C_p
        self.C_cur = C_cur

        self.cells = nn.ModuleList()
        reduction_p = False
        duplicate = False
        reduce_location = eval(config.reduce_loc)

        if reduce_location[1] > 2*n_layers//3:
            raise ValueError("second index of reduce location should be small or equal than 2*n_layers//3")

        for i in range(n_layers):
            # if i in [reduce_location[0]*n_layers//6, reduce_location[1]*n_layers//6]:
            if i in reduce_location:
                if self.config.no_balancing:
                    C_cur *= 2
                else:
                    C_cur *= 4
                reduction = True
            else: 
                reduction = False

            cell = AugmentCell(config, genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur
            C_pp, C_p = C_p, C_cur_out

            if i == self.aux_pos:
                if config.dataset == 'tiny_imagenet':
                    self.aux_head = AuxiliaryHeadTinyImageNet(input_size // 4, C_p, n_classes)
                else:
                    self.aux_head = AuxiliaryHead(input_size//4, C_p, n_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        # print("s0 shape after stem:/ ", s0.shape)
        # print("s1 shape after stem: ", s1.shape)
        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)
                
        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits, aux_logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
