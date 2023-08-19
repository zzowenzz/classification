import torch

def get_device(i=0): 
    '''
    Train network on gpu or cpu
    '''
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')