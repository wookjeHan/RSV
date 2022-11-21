import os
import random
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

def topk_filter(tensor, k):
    if k == 0:
        return tensor

    topk, _ = torch.topk(tensor, k=k)
    thres = topk[:,:,-1].unsqueeze(-1)
    return torch.where(tensor < thres, torch.full_like(tensor, float('-inf')), tensor)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_exp_name(args, trim=False):
    exp_name = f"sn{args.shot_num}_k{args.topk}_lr{args.lr}"

    if args.weight_decay > 0.0:
        exp_name += f"_wd{args.weight_decay}"

    if args.replace:
        exp_name += f"_rep"

    if args.tv_split_ratio == 0.0:
        exp_name += f"_endo"

    if trim:
        return exp_name
    else:
        return os.path.join(
            'OURS',
            f"{args.dataset}_sr{args.tv_split_ratio}",
            args.prompt,
            exp_name
        )

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[97m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'