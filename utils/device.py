import torch

def get_device(num_gpu=1):
    '''
    Train network on multi GPUs, single GPU, or CPU
    '''
    available_gpus = torch.cuda.device_count()
    
    if available_gpus >= num_gpu and num_gpu > 1:
        return [torch.device(f'cuda:{i}') for i in range(num_gpu)]
    elif available_gpus >= 1:
        return [torch.device('cuda:0')]
    else:
        return [torch.device('cpu')]
