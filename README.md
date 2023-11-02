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



# Reference
1. https://zh.d2l.ai/index.html
2. https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification