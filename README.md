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

## TODO 
1. [x] git init commit
2. [x] use torchinfo to save architecture
3. [x] build seed function
4. [x] build device function 
5. [x] build dataloader and data transform function
6. [x] build logger in log/train.log
7. [x] load pretrain weight
8. [x] check pretrain weight loading and claim in loggin
9. [x] load default cfg and use customied yaml files to overwrite the default
10. [x] move every setting to cfg
11. [ ] build different warm up scheduler
12. [x] use __main__ to run the code
13. [x] use keep main() and train() in train.py
14. [x] log to console and file 
15. [x] train in DataParallel
16. [x] sync training with wandb
17. [x] create model folder and separate each model
18. [x] Lenet, Alexnet, VGG
19. [x] ViT 


# Reference
1. https://zh.d2l.ai/index.html
2. https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification