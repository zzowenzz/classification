# Pytorch official ddp demo
import torch
import torch.distributed as dist
import torch.multiprocessing as mp # python mp tailored for pytorch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP # wrapper that parallelizes model for data parallel training

import os

def example(rank, world_size):
    # create default process group; a group of processes that communicate with each other for dist training
    # each process has a unique identifier called rank (integer, from 0 to world_size -1) and world_size (total num of processes)
    # process group has many communicatation operation:
        # broadcast: send data from 1 to all other processes, makes everyone have the same data
        # reduce: aggregate data from all processes to 1
        # all-reduce: aggregate data from all processes and distribute the result to all processes
        # gather: gather data from all processes to 1
        # scatter: distribute (different but equal-sized) data from 1 to all processes
        # barrier: synchronize all processes
    # "map" and "reduce" come from functional programming where "map" applies a function to each element in a list 
    # and "reduce" aggregates all elements in a list to a single value
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # create local model
    model = nn.Linear(10, 10).to(rank)

    # wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2 # num of processes in DDP mode; equals to num of GPUs; 1 process = 1 GPU
    mp.spawn( # start the multi processes
        example, # function to be run
        args=(world_size,), # arguments to be passed to the function
        nprocs=world_size, # num of processes to spawn, equal to world_size; each process will run the function
        join=True)

if __name__=="__main__":
    # The MASTER_ADDR and MASTER_PORT is a common meeting point for all these processes 
    # to synchronize and establish communication.
    os.environ["MASTER_ADDR"] = "localhost" # network address of the master process; set to localhost for single machine multi gpu
    os.environ["MASTER_PORT"] = "29500" # master port where the processes rendezvous(meet; 
    # rendezvous is for an initial synchronization step where all processes discover each other 
    # and agree on how to communicate
    main()