import torchvision
from torchvision import transforms
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

import multiprocessing

def get_transforms(augment=False):
    '''
    Return different transformations
    '''
    base_transforms = [transforms.ToTensor()]
    if augment:
        data_augmentation_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # Add more augmentation transforms if needed

        ]
        base_transforms = data_augmentation_transforms + base_transforms
    return transforms.Compose(base_transforms)

def get_dataloader(batch_size):
    '''
    Load data
    '''
    trans = {
        "train": get_transforms(),
        "test": get_transforms(),
    }
    train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans["train"], download=True)
    train_sampler = DistributedSampler(train)# wrap train dataset with DistributedSampler
    test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans["test"], download=True)
    return (data.DataLoader(train, batch_size, shuffle=False, num_workers=multiprocessing.cpu_count(), sampler=train_sampler),
            data.DataLoader(test, batch_size, shuffle=False,num_workers=multiprocessing.cpu_count()),
            train_sampler) # return train_sampler for differently shuffling in each epoch