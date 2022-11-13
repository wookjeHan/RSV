import numpy as np
import torch

_use_gpu = False
device = None

def set_device(target_device):
    global _use_gpu
    global device
    _use_gpu = 'cuda' in target_device
    device = torch.device(target_device)

def get_device():
    return device

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)
