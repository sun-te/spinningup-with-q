import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from ipdb import set_trace as tt

class LinearScheduler:
    def __init__(self, total_step, init_value, final_value):
        self.total_step = total_step
        self.init_value = init_value
        self.final_value = final_value

    def value(self, t):
        v = t * 1.0 /self.total_step * (self.init_value - self.final_value) + self.init_value
        ans = min(1.0, v)
        return ans

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):

        q = self.q(obs)

        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

