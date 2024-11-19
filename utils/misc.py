import os
import torch
import random
import numpy as np
from omegaconf import DictConfig
from typing import Dict

def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    if isinstance(x, torch.Tensor):
        x = x.to(dtype=dtype, device=device)
        x = x.clone().detach().requires_grad_(requires_grad)
    else:
        x = torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
    return x

def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def set_np_formatting():
    """formats numpy print"""
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return seed


class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = -np.inf

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        if self.mean == -np.inf:
            self.mean = 0
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = -np.inf

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean
