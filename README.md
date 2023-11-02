# FashionMNIST Image Classification with Famous Networks
This repository provides code and instructions for training popular deep learning networks on the FashionMNIST dataset for image classification.

## Overview
FashionMNIST is a dataset of Zalando's article images, consisting of 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

n this project, I aim to replicate the training of famous networks such as VGG, ResNet, MobileNet, Vision Transformer on the FashionMNIST dataset.
## Installation
```
# create conda environment
conda create -n cls python=3.10
conda activate cls

# clone this repo
git clone https://github.com/zzowenzz/classification.git
# install required libraries
sh requirements.sh
```


## Train
Training command is for Pytorch distributed training on Linux.
```
torchrun --nnode=[NUMBER OF NODES] --nproc_per_node=[NUMBER OF GPU PER NODE] --node_rank=[RANK OF THIS NODE] train.py --cfg [CONFIG FILE] --WANDB [WANDB KEY]
```

## Paper
| Network | Paper | 
| :---: | :---: |
| LeNet | [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791) | 
| AlexNet | [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) |
| VGG | [VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)|
| ResNet | [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) |
| ViT | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf?fbclid=IwAR1NafJDhZjkARvCswpV6kS9_hMa0ycvzwhlCb7cqAGwgzComFXcScxgA8o) |

# Reference
1. https://zh.d2l.ai/index.html
2. https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification