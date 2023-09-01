import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn as nn

import pandas as pd
import time
import multiprocessing
import os 
import logging
import datetime
import argparse

from model import Lenet
from utils.seed import set_seed
from utils.device import get_device
from utils.dataloader import get_dataloader
from utils.load_pretrain import load_pretrain
from config.default import cfg

# set arguments
parser = argparse.ArgumentParser(description='Train LeNet on FashionMNIST')
parser.add_argument('--cfg', type=str, default='',help='configuration of training')
args = parser.parse_args()

# set logger
current_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
log_filename = f'./log/{current_time}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# load config
cfg.merge_from_file(args.cfg)
logging.info(cfg)

# Prepare all parts: network, device, data iterator, network initialization, optimizer, loss function
net = Lenet()
net_name = os.getcwd().split("/")[-1]
device = get_device()
train_iter, test_iter, num_train, num_test = get_dataloader(cfg.TRAIN.BATCH_SIZE)
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)
initial_weights = {name: param.clone() for name, param in net.named_parameters()}

pretrain = cfg.BACKBONE.PRETRAINED
if pretrain:
    load_pretrain(net, pretrain)

net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=cfg.TRAIN.LR)
loss = nn.CrossEntropyLoss()

# set seed to reproduce result
set_seed(0)

# Train
logging.info("Train {} on {}".format(net_name, device))
logging.info("{} images for training, {} images for validation\n".format(num_train,num_test))
total_time = 0.0
for epoch in range(cfg.TRAIN.EPOCHS):
    batch_time = time.time()
    train_loss, train_acc, test_acc, best_acc = 0.0, 0.0, 0.0, 0.0
    net.train()
    # For each batch 
    for i, (x, y) in enumerate(train_iter):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
    
        with torch.no_grad():
            train_loss += l/cfg.TRAIN.BATCH_SIZE
            train_acc += (y_hat.argmax(axis=1) == y).sum().item()
    train_acc /= num_train

    net.eval()
    with torch.no_grad():
        # For each batch
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            test_acc += (net(X).argmax(axis=1) == y).sum().item()
    test_acc /= num_test
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(net.state_dict(), cfg.TRAIN.SNAPSHOT_BEST)
    time_end = time.time()
    total_time += (time_end - batch_time)
    logging.info("Epoch {}, train_loss {}, train_acc {}, best_acc {}, test_acc {}, time cost {} sec".format(epoch+1, "%.4f" % train_loss, "%.2f" % train_acc, "%.2f" %best_acc, "%.2f" %test_acc,  "%.2f" %(time_end - batch_time)))

logging.info("\nFinish training")

