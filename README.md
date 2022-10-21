##  SPHYNX: A Deep Neural Network Design for Private Inference - Published in [IEEE Privacy & Security](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9786822)

#### Authors: Minsu Cho, Zahra Ghodsi, Brandon Reagen, Siddharth Garg, and Chinmay Hegde

NAS Framework on micro-search space, enabling efficient private inference.

### Setup

To setup the environment, use the requirements.txt file. 

Basic requirements:

1. Pytorch == 1.5.0
2. Torchvision == 0.6.0
3. Numpy == 1.18.4


### Codes

We include the codes for experiments conducted in the papers as following:

`search_concat.py`: Finding normal/reduce cells from the SPHYNX search space.

`search_concat_loc.py`: Finding reduce cells locations.

`evaluation/augment_concat.py`: Training final network for CIFAR-100 and Tiny-ImageNet

`evaluation/augment_concat.py`: Training final network for ImageNet.

### Experiments

#### CIFAR-100

SPHYNX-ReLU 25K: `python ./evaluation/augment_concat.py --name "C100_SPHYNX_ReLU25K" --dataset "cifar100" --init_channels 5 --layers 5 --reduce_loc "[0,1]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))"`

SPHYNX-ReLU 30K: `python ./evaluation/augment_concat.py --name "C100_SPHYNX_ReLU30K" --dataset "cifar100" --init_channels 5 --layers 6 --reduce_loc "[0,1]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))"`

SPHYNX-ReLU 40K: `python ./evaluation/augment_concat.py --name "C100_SPHYNX_ReLU40K" --dataset "cifar100" --init_channels 5 --layers 8 --reduce_loc "[1,3]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))"`

SPHYNX-ReLU 50K: `python ./evaluation/augment_concat.py --name "C100_SPHYNX_ReLU50K" --dataset "cifar100" --init_channels 5 --layers 10 --reduce_loc "[0,5]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))`

SPHYNX-ReLU 70K: `python ./evaluation/augment_concat.py --name "C100_SPHYNX_ReLU70K" --dataset "cifar100" --init_channels 7 --layers 10 --reduce_loc "[0,5]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))`

SPHYNX-ReLU 100K: `python ./evaluation/augment_concat.py --name "C100_SPHYNX_ReLU100K" --dataset "cifar100" --init_channels 10 --layers 10 --reduce_loc "[0,5]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))`

#### Tiny-ImageNet

SPHYNX-ReLU 102K: `python ./evaluation/augment_concat.py --name "Tiny_SPHYNX_ReLU102K" --dataset "tiny_imagenet" --init_channels 5 --layers 5 --reduce_loc "[0,1]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))"`

SPHYNX-ReLU 204K: `python ./evaluation/augment_concat.py --name "Tiny_SPHYNX_ReLU204K" --dataset "tiny_imagenet" --init_channels 5 --layers 10 --reduce_loc "[0,5]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))"`

SPHYNX-ReLU 286K: `python ./evaluation/augment_concat.py --name "Tiny_SPHYNX_ReLU286K" --dataset "tiny_imagenet" --init_channels 7 --layers 10 --reduce_loc "[0,5]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))"`

SPHYNX-ReLU 492K: `python ./evaluation/augment_concat.py --name "Tiny_SPHYNX_ReLU492K" --dataset "tiny_imagenet" --init_channels 12 --layers 10 --reduce_loc "[0,5]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))"`

SPHYNX-ReLU 614K: `python ./evaluation/augment_concat.py --name "Tiny_SPHYNX_ReLU614K" --dataset "tiny_imagenet" --init_channels 15 --layers 10 --reduce_loc "[0,5]" --drop_path_prob 0.2 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))"`

#### ImageNet

SPHYNX-ReLU 345K: `python ./evaluation/augment_concat_imagenet.py --name "ImageNet_SPHYNX_ReLU345K" --dataset "imagenet" --init_channels 20 --layers 10 --reduce_loc "[1,5]" --drop_path_prob 0.0 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))" --workers 20 --imagenet_stem_relu --imagenet_train_path "imagenet/dir" --imagenet_valid_path "imagenet/dir" --epochs 120 --lr 0.1 --batch_size 768 --label_smooth 0.0` 

SPHYNX-ReLU 517K: `python ./evaluation/augment_concat_imagenet.py --name "ImageNet_SPHYNX_ReLU517K" --dataset "imagenet" --init_channels 30 --layers 10 --reduce_loc "[1,5]" --drop_path_prob 0.0 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))" --workers 20 --imagenet_stem_relu --imagenet_train_path "imagenet/dir" --imagenet_valid_path "imagenet/dir" --epochs 120 --lr 0.1 --batch_size 768 --label_smooth 0.0`

SPHYNX-ReLU 690K: `python ./evaluation/augment_concat_imagenet.py --name "ImageNet_SPHYNX_ReLU690K" --dataset "imagenet" --init_channels 40 --layers 10 --reduce_loc "[1,5]" --drop_path_prob 0.0 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))" --workers 20 --imagenet_stem_relu --imagenet_train_path "imagenet/dir" --imagenet_valid_path "imagenet/dir" --epochs 120 --lr 0.1 --batch_size 768 --label_smooth 0.0`

SPHYNX-ReLU 862K: `python ./evaluation/augment_concat_imagenet.py --name "ImageNet_SPHYNX_ReLU862K" --dataset "imagenet" --init_channels 50 --layers 10 --reduce_loc "[1,5]" --drop_path_prob 0.0 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))" --workers 20 --imagenet_stem_relu --imagenet_train_path "imagenet/dir" --imagenet_valid_path "imagenet/dir" --epochs 120 --lr 0.1 --batch_size 768 --label_smooth 0.0`

SPHYNX-ReLU 1034K: `python ./evaluation/augment_concat_imagenet.py --name "ImageNet_SPHYNX_ReLU1034K" --dataset "imagenet" --init_channels 60 --layers 10 --reduce_loc "[1,5]" --drop_path_prob 0.0 --genotype "Genotype(normal=[[('skip_connect_NR', 0), ('skip_connect_NR', 1)], [('skip_connect_NR', 2), ('van_dil_conv_5x5_NR', 0)], [('van_conv_5x5_NR', 2), ('van_dil_conv_3x3_NR', 0)], [('van_conv_3x3_NR', 1), ('skip_connect_NR', 2)]], normal_concat=range(2, 6), reduce=[[('van_conv_3x3_NR', 0), ('van_conv_3x3_NR', 1)], [('skip_connect_NR', 0), ('van_conv_3x3_NR', 2)], [('van_dil_conv_3x3_NR', 0), ('skip_connect_NR', 2)], [('van_conv_5x5_NR', 0), ('skip_connect_NR', 4)]], reduce_concat=range(2, 6))" --workers 20 --imagenet_stem_relu --imagenet_train_path "imagenet/dir" --imagenet_valid_path "imagenet/dir" --epochs 120 --lr 0.1 --batch_size 768 --label_smooth 0.0`
