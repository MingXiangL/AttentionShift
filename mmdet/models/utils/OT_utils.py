import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_distance(x, y):
    x_norm = x.norm(p=2, dim=-1, keepdim=True)
    y_norm = y.norm(p=2, dim=-1, keepdim=True).clamp(1e-6)
    sim = torch.matmul(x / x_norm, (y / y_norm).T)
    return 1 - sim, sim

def cosine_distance_part(x, list_y):
    split_y = [y_i.shape[0] for y_i in list_y]
    cost = cosine_distance(x, torch.cat(list_y)).mean(0)
    return cost.split(split_y, dim=1)