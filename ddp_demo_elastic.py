# Pytorch official ddp elastic demo
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    # initializes the distributed environment with the "nccl" backend
    dist.init_process_group("nccl")

    rank = dist.get_rank() # get the currect rank inside the dist process group; default is 0

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    print(device_id)
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    print("Dist rank: {}, loss: {}".format(rank, loss_fn(outputs, labels).item()))
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()

# torchrun 
# --nnodes=1  (num of machine)
# --nproc_per_node=2  (num of gpu/process)
# --rdzv_id=100 # 
# --rdzv_backend=static (single machine multi gpu where num of nodes and processes per node are fixed); can also be "etcd" (elastic training where num of nodes or gpu can change during training) and "c10d" (multi nodes)
# --rdzv_endpoint=localhost:29400 (address of the rendezvous server; set to master node for c10d; set to etcd server for etcd)
# ddp_demo_elastic.py 