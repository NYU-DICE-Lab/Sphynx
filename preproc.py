import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def data_transforms(dataset, cutout_length):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'cifar100':
        MEAN = [0.5071, 0.4867, 0.4408]
        STD = [0.2675, 0.2565, 0.2761]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'mnist':
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
    elif dataset == 'fashionmnist':
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    elif dataset == 'tiny_imagenet':
        MEAN = [0.480, 0.448, 0.398]
        STD = [0.277, 0.269, 0.282]
        transf = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip()
        ]    
    elif dataset == 'imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        transf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ]
        transf2 = [
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    if dataset == 'imagenet':
        train_transform = transforms.Compose(transf + normalize)
        valid_transform = transforms.Compose(transf2 + normalize)
    else:   
        train_transform = transforms.Compose(transf + normalize)
        valid_transform = transforms.Compose(normalize)

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform
