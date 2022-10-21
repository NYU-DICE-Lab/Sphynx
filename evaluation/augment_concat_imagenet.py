""" Training augmented model for crytponas
    Adding the relu in the beginning but separating out... 
"""
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from configs.config_concat import AugmentConfig
import utils
from models_concat.augment_cnn import AugmentCNN, AugmentCNNImageNet


config = AugmentConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)


logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

n_classes = 1000

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def main():
    # accuracy_list = []
    for _ in range(1):
        logger.info("Logger is set - training start")

        # set default gpu device id
        # torch.cuda.set_device(config.gpus[0])

        n_gpus_per_node = torch.cuda.device_count()
        gpu = None

        # set seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        torch.backends.cudnn.benchmark = True

        # get data with meta info
        input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
            config, config.dataset, config.data_path, config.cutout_length, validation=True)

        criterion = nn.CrossEntropyLoss().to(device)
        criterion_smooth = CrossEntropyLabelSmooth(n_classes, config.label_smooth).to(device)

        use_aux = config.aux_weight > 0.
        model = AugmentCNNImageNet(input_size, input_channels, config.init_channels, n_classes, config.layers,
                        use_aux, config.genotype, config)
        # model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                        # use_aux, config.genotype, config)
        # model = nn.DataParallel(model, device_ids=config.gpus).to(device)
        model = nn.DataParallel(model).to(device)
        # model size
        mb_params = utils.param_size(model)
        logger.info("Model size = {:.3f} MB".format(mb_params))

        # weights optimizer
        optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)

        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=config.batch_size,
                                                shuffle=True,
                                                num_workers=config.workers,
                                                pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_data,
                                                batch_size=config.batch_size,
                                                shuffle=False,
                                                num_workers=config.workers,
                                                pin_memory=True)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.epochs))

        # Resuming the model if possible.
        if os.path.isfile(os.path.join(config.path, "checkpoint.pth.tar")):
            print("=> loading checkpoint {}".format(os.path.join(config.path, "checkpoint.pth.tar")))
            checkpoint = torch.load(os.path.join(config.path, "checkpoint.pth.tar"))
            start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            model.load_state_dict(checkpoint['state_dict'])
            print("Resume Successful!")
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(os.path.isfile(os.path.join(config.path, "checkpoint.pth.tar")), checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(os.path.isfile(os.path.join(config.path, "checkpoint.pth.tar"))))
            start_epoch = 0
            best_top1 = 0.

        # training loop
        for epoch in range(start_epoch, config.epochs):
            current_lr = adjust_lr(optimizer, epoch)
            
            # lr_scheduler.step()
            drop_prob = config.drop_path_prob * epoch / config.epochs
            model.module.drop_path_prob(drop_prob)

            # training
            if config.label_smooth > 0:
                train(train_loader, model, optimizer, criterion_smooth, epoch)
            else:
                train(train_loader, model, optimizer, criterion, epoch)

            # validation
            cur_step = (epoch+1) * len(train_loader)
            top1 = validate(valid_loader, model, criterion, epoch, cur_step)

            # save
            if best_top1 < top1:
                best_top1 = top1
                is_best = True
            else:
                is_best = False

            utils.save_checkpoint_imagenet(
                config, {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_top1': best_top1,
                    'optimizer': optimizer.state_dict(),
                }, is_best
            )

            # utils.save_checkpoint(model, config.path, is_best)
            # lr_scheduler.step()


            print("")

        logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
        # accuracy_list.append(best_top1)
        # print("Final accuracy list: ", accuracy_list)


def train(train_loader, model, optimizer, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg

# def adjust_lr(optimizer, epoch):
#     if config.epochs - epoch > 5:
#         lr = config.lr * (config.epochs - 5 - epoch) / (config.epochs - 5)
#     else:
#         lr = config.lr * (config.epochs - epoch) / ((config.epochs - 5) * 5)

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

# Using the regular step function style 
def adjust_lr(optimizer, epoch):
    if epoch < 30:
        lr = config.lr
    elif epoch >= 30 and epoch < 60:
        lr = config.lr * 0.1
    elif epoch >= 60 and epoch < 90:
        lr = config.lr * 0.01
    else: 
        lr = config.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == "__main__":
    main()
