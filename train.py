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

from model import Lenet
from utils.seed import set_seed
from utils.device import get_device
from utils.dataloader import get_dataloader
from utils.load_pretrain import load_pretrain

# set logger
current_time = datetime.datetime.now().strftime('%m_%d_%H_%M')
log_filename = f'./log/{current_time}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Hyper-parameter: batch_size, lr, num_epochs
batch_size = 256
lr, num_epochs = 0.1, 10
best_acc, save_path = 0.0, "best.pth"

logging.info("batch_size: {}, lr: {}, num_epochs: {}".format(batch_size, lr, num_epochs))

# Prepare all parts: network, device, data iterator, network initialization, optimizer, loss function
net = Lenet()
net_name = os.getcwd().split("/")[-1]
device = get_device()
train_iter, test_iter, num_train, num_test = get_dataloader(batch_size)
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)
initial_weights = {name: param.clone() for name, param in net.named_parameters()}

pretrain = True
pretrain_path = "./best.pth"
if pretrain:
    load_pretrain(net, pretrain_path)

net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

# set seed
set_seed(0)
# loss will be same for unshuffled dataset


# Train
# print("Train {} on {}".format(net_name, device))
logging.info("Train {} on {}".format(net_name, device))
# print("{} images for training, {} images for validation\n".format(num_train,num_test))
logging.info("{} images for training, {} images for validation\n".format(num_train,num_test))
total_time = 0.0
for epoch in range(num_epochs):
    batch_time = time.time()
    train_loss, train_acc, test_acc = 0.0, 0.0, 0.0
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
            train_loss += l/batch_size
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
        torch.save(net.state_dict(), save_path)
    time_end = time.time()
    total_time += (time_end - batch_time)
    # print("Epoch {}, train_loss {}, train_acc {}, best_acc {}, test_acc {}, time cost {} sec".format(epoch+1, "%.4f" % train_loss, "%.2f" % train_acc, "%.2f" %best_acc, "%.2f" %test_acc,  "%.2f" %(time_end - batch_time)))
    logging.info("Epoch {}, train_loss {}, train_acc {}, best_acc {}, test_acc {}, time cost {} sec".format(epoch+1, "%.4f" % train_loss, "%.2f" % train_acc, "%.2f" %best_acc, "%.2f" %test_acc,  "%.2f" %(time_end - batch_time)))
# number of parameter
with open("architecture.txt", "r") as f:
    for line in f:
        if "Total params: " in line:
            num_para = line.split()[-1]
# df = pd.concat([df, pd.DataFrame.from_records([{"Network":net_name, "Parameter": num_para, "Dataset":"FashionMNIST", "Epoch":num_epochs, "Device":torch.cuda.get_device_name(0), "Time cost(sec)": "%.1f" %total_time , "Batch size":batch_size, "Lr":lr, "Best test acc":best_acc}])])
# df.to_csv(net_name+".csv",index=False,header=True)
# print("\nFinish training")
logging.info("\nFinish training")

