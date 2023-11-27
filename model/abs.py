import os, sys
import torch
from torch import nn


class Core(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = None
        self.decoder = None
    
    def forward(self, x):
        raise NotImplementedError()
    
    def get_share_params(self):
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        self.encoder.zero_grad()