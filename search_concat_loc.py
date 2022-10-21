"""
This is creating the multiple branches of the 
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from configs.config_concat_loc import SearchConfig
import utils
from models_concat.search_cnn import SearchCNNController
from models_concat.augment_cells import AugmentCell_NoPrep, AugmentCell_Preproc1, AugmentCell, AugmentCell_NoReLU
from models_concat import ops

import random

config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


class LocationCNN(nn.Module):
    """
    CNN with multiple different cells in each locations.  
    """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, genotype, config, stem_multiplier=3):
        """
        Args: 
            input_size: size of height and width.
            C_in: # of input channels
            C: initial number of channels.
        """
        super().__init__()
        self.C_in = C_in
        self.C = C 
        self.n_classes = n_classes 
        self.n_layers = n_layers 
        self.genotype = genotype  
        self.config = config

        C_cur = C * stem_multiplier      # This is to make things convinient

        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        C_pp, C_p, C_cur = C_cur, C_cur, C 

        # saving this for the forward pass
        self.C_pp = C_pp
        self.C_p = C_p 
        self.C_cur = C_cur 

        # all possible pair list given the number of layers
        self.reduce_pair_list = []
        for i in range(2*self.n_layers//3+1):
            for j in range(i):
                self.reduce_pair_list.append((j, i))

        self.skeleton = nn.ModuleList()

        print("reduce_pair_list: ", self.reduce_pair_list)

        for k, reduce_pair in enumerate(self.reduce_pair_list):
            print("k: {} and reduce_pair: {}".format(k, reduce_pair))

            sub_skeleton = nn.ModuleList()
            reduction_p = False

            # reseting the channel size
            C_pp, C_p, C_cur = self.C_pp, self.C_p, self.C_cur

            for i in range(self.n_layers):
                if i in reduce_pair:
                    C_cur *= 4
                    reduction = True
                else:
                    reduction = False

                cell = AugmentCell(config, genotype, C_pp, C_p, C_cur, reduction_p, reduction)
                reduction_p = reduction 
                sub_skeleton.append(cell)
                C_cur_out = C_cur
                C_pp, C_p = C_p, C_cur_out
            
            sub_skeleton.append(nn.AdaptiveAvgPool2d(1))
            sub_skeleton.append(nn.Linear(C_p, n_classes))
            self.skeleton.append(sub_skeleton)

        print("Skeleton length: ", len(self.skeleton))
        # Beta location: 
        self.beta = nn.ParameterList()
        self.beta.append(nn.Parameter(1e-3*torch.randn(1, len(self.skeleton))))

        self._betas = []
        for n, p in self.named_parameters():
            if 'beta' in n:
                self._betas.append((n, p))

        self.tau = 10


    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def get_beta(self):
        return self.beta

    def betas(self):
        for n, p in self._betas:
            yield p

    def get_beta_softmax(self):
        logits = self.beta[0].log_softmax(dim=1) 
        beta_softmax = nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        return beta_softmax

    def forward(self, x):

        def get_gumble_prob(xins):
            while True:
                gumbels = -torch.empty_like(xins).exponential_().log()
                logits = (xins.log_softmax(dim=1) + gumbels) / self.tau
                probs = nn.functional.softmax(logits, dim=1)
                index = probs.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if torch.isinf(gumbels).any() or torch.isinf(probs).any() or torch.isnan(probs).any():
                    continue
                else:
                    break

            return hardwts, index


        # First option here
        # one_hot, index = get_gumble_prob(self.beta[0])
        
        # states = []
        # for i, sub_network in enumerate(self.skeleton):
        #     s0 = s1 = self.stem(x)
        #     for j, cell in enumerate(sub_network):
        #         s0, s1 = s1, cell(s0, s1)
        #         if j == len(sub_network)-3:
        #             break
        #     out = sub_network[len(sub_network)-2](s1)
        #     out = out.view(out.size(0), -1)   # flatten
        #     logits = sub_network[len(sub_network)-1](out)
        #     states.append(logits)

        # weighted_logit_sum = sum(state * w for state, w in zip(states, one_hot[0]))
        # states[index]

        # return weighted_logit_sum
        
        # Second option here. 
        # one_hot, index = get_gumble_prob(self.beta[0])
        # sub_network = self.skeleton[index[0]]
        # s0 = s1 = self.stem(x)
        # for i, cell in enumerate(sub_network):
        #     s0, s1 = s1, cell(s0, s1)
        #     if i == len(sub_network)-3:
        #         break
        # out = sub_network[len(sub_network)-2](s1)
        # out = out.view(out.size(0), -1)
        # logits = one_hot[0][index[0]]*sub_network[len(sub_network)-1](out)

        # return logits

        # Third option here. First option but subsampling.
        one_hot, index = get_gumble_prob(self.beta[0])
        # print(index.item())
        # print(self.skeleton[index.item()])
        if self.config.subsampling == 0:
            sub_index = [index.item()]
        else:
            while True:
                sub_index = random.sample(range(len(one_hot[0])), 3)
                if index.item() not in sub_index:
                    continue
                else:
                    break
                
        sub_index.sort()
        # print(sub_index)
        states = []
        for i in (sub_index):
            s0 = s1 = self.stem(x)
            for j, cell in enumerate(self.skeleton[i]):
                s0, s1 = s1, cell(s0, s1)
                if j == len(self.skeleton[i])-3:
                    break
            out = self.skeleton[i][len(self.skeleton[i])-2](s1)
            out = out.view(out.size(0), -1)   # flatten
            logits = self.skeleton[i][len(self.skeleton[i])-1](out)
            states.append(logits)
        
        weighted_logit_sum = sum(state * one_hot[0][ind] for state, ind in zip(states, sub_index))
        return weighted_logit_sum

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            # for module in sub_modules:
            if isinstance(module, ops.DropPath_):
                module.p = p

def main(config):
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config, config.dataset, config.data_path, cutout_length=0, validation=False)

    criterion = nn.CrossEntropyLoss().to(device)
    # use_aux = config.aux_weight > 0.
    # This needs to be changed. 
    # model = SearchCNNController(config, input_channels, config.init_channels, n_classes, config.layers,
                                # net_crit, device_ids=config.gpus)


    print("Reading the LocationCNN model")
    model = LocationCNN(input_size, input_channels, config.init_channels, n_classes, config.layers, config.genotype, config)
    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.parameters(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # beta optimizer
    beta_optim = torch.optim.Adam(model.betas(), config.beta_lr, betas=(0.5, 0.999), weight_decay=config.beta_weight_decay)


    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)

    # training loop
    best_top1 = 0.

    logger.info("self.reduce_pair_list: {}".format(model.reduce_pair_list))
    logger.info("Beta softmax: {}".format(nn.functional.softmax(model.beta[0].data, dim=1).detach().cpu().numpy()))
    for epoch in range(config.epochs):
        lr = lr_scheduler.get_lr()[0]

        # Gradually tau gets smaller
        model.set_tau(config.tau_max-(config.tau_max-config.tau_min) * epoch / (config.epochs-1))
        
        # drop_prob = config.drop_path_prob * epoch / config.epochs
        # model.drop_path_prob(drop_prob)
        
        print("Epochs: [{:2d}/{}] Tau {}".format(epoch+1, config.epochs, model.get_tau()))
        # training
        train(train_loader, valid_loader, model, criterion, w_optim, beta_optim, lr, epoch, config)
        # validation
        cur_step = (epoch+1) * len(train_loader)
        # top1 = validate(valid_loader, model, criterion, epoch, cur_step, config)

        lr_scheduler.step()
        print("self.reduce_pair_list: ", model.reduce_pair_list)
        # print("Beta softmax: ", nn.functional.softmax(model.beta[0].data, dim=1))
        logger.info("self.reduce_pair_list: {}".format(model.reduce_pair_list))
        logger.info("Beta softmax: {}".format(nn.functional.softmax(model.beta[0].data, dim=1).detach().cpu().numpy()))
        # log
        # genotype
        # genotype = model.genotype()
        # logger.info("genotype = {}".format(genotype))

        # # genotype as a image
        # plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        # caption = "Epoch {}".format(epoch+1)
        # plot(genotype.normal, plot_path + "-normal", caption)
        # plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        # if best_top1 < top1:
        #     best_top1 = top1
        #     best_genotype = genotype
        #     is_best = True
        # else:
        #     is_best = False
        # utils.save_checkpoint(model, config.path, is_best)
        print("")


    print(model.get_beta_softmax())
    # logger.info("Final beta softmax = {:.4%}".format(model.get_beta_softmax()))
    logger.info("Final beta logits = {}".format(np.array2string(model.get_beta_softmax(), precision=5, separator=',', suppress_small=True)))
    # logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    # logger.info("Best Genotype = {}".format(best_genotype))

def train(train_loader, valid_loader, model, criterion, w_optim, beta_optim, lr, epoch, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
     

        N = trn_X.size(0)

        # # phase 2. updating the architecture parameter.
        beta_optim.zero_grad()     
        logits = model(val_X)
        # print("logits in cuda? ", logits.is_cuda)
        loss = criterion(logits, val_y)
        loss.backward()
        beta_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info("Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

def validate(valid_loader, model, criterion, epoch, cur_step, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % 100 == 0 or step == len(valid_loader) - 1:
                logger.info("Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

            

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg

if __name__ == "__main__":
    main(config)
