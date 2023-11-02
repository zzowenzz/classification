import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import pandas as pd
import time
import multiprocessing
import os 
import logging
import datetime
import argparse
import wandb

from model.resnext import resnext50_2x40d # import model. You many need to change this line
from utils.seed import set_seed
from utils.device import get_device
from utils.dataloader import get_dataloader
from utils.load_pretrain import load_pretrain
from config.default import cfg
from utils.cleanup import cleanup
from utils.log import log_from_gpu

def set_dist(net):
    # init
    dist.init_process_group("nccl")
    rank = dist.get_rank() # get the currect rank inside the dist process group; default is 0
    device_id = rank % torch.cuda.device_count()

    # wrap the model with DDP
    net = net.to(device_id)
    net = DDP(net, device_ids=[device_id])
    return net, device_id

def train(train_iter, test_iter, train_sampler, net, loss, optimizer, device):
    # Train
    total_time = 0.0
    for epoch in range(cfg.TRAIN.EPOCHS):
        train_sampler.set_epoch(epoch) # set the epoch for the sampler for different shuffling in each epoch
        batch_time = time.time()
        train_loss, train_acc, test_acc, best_acc = 0.0, 0.0, 0.0, 0.0
        net.train()
        # For each batch 
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += l/cfg.TRAIN.BATCH_SIZE
                train_acc += (y_hat.argmax(axis=1) == y).sum().item()
                train_acc /= X.shape[0]

        net.eval()
        with torch.no_grad():
            # For each batch
            for X, y in test_iter:
                X = X.to(device)
                y = y.to(device)
                test_acc += (net(X).argmax(axis=1) == y).sum().item()
                test_acc /= X.shape[0]
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), cfg.TRAIN.SNAPSHOT_BEST)
        time_end = time.time()
        total_time += (time_end - batch_time)
        log_from_gpu("Epoch {}, train_loss {}, train_acc {}, best_acc {}, test_acc {}, time cost {} sec".format(epoch+1, "%.4f" % train_loss, "%.2f" % train_acc, "%.2f" %best_acc, "%.2f" %test_acc,  "%.2f" %(time_end - batch_time)))
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "best_acc": best_acc, "test_acc": test_acc, "time_cost": time_end - batch_time}) if args.wandb else None

    log_from_gpu("\nFinish training")

def main(args):
    # create logger
    current_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
    if not os.path.exists("./log/"+current_time):
        os.makedirs("./log/"+current_time)
    log_filename = f'./log/{current_time}/{current_time}.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    
    # load cfg
    cfg.merge_from_file(args.cfg)
    log_from_gpu(cfg)

    # create model
    net = resnext50_2x40d() # you may need to change this line, according to your imported model
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # set dist training
    net, device = set_dist(net)

    # load pretrain model
    if cfg.BACKBONE.PRETRAINED:
        load_pretrain(net, cfg.BACKBONE.PRETRAINED)

    # create training recorder
    if args.wandb:
        wandb.init(project="classification", name=f"{current_time}")

    # build dataloader
    train_iter, test_iter, train_sampler = get_dataloader(cfg.TRAIN.BATCH_SIZE)
    
    # create loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.TRAIN.LR)

    # start training
    train(train_iter, test_iter, train_sampler, net, loss, optimizer, device)

if __name__ == '__main__':
    # set seed to reproduce result
    set_seed(0)

    # set arguments
    parser = argparse.ArgumentParser(description='Train on FashionMNIST')
    parser.add_argument('--cfg', type=str, default='',help='configuration of training')
    parser.add_argument("--wandb", type=str, required=False, help="API key for wandb.")
    args = parser.parse_args()

    main(args)
    cleanup() # clean up for dist training

