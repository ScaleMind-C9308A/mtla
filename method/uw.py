import os, sys
from typing import *

import torch
from torch import nn
from .core import Vanilla 


class UW(Vanilla):
    def __init__(self):
        super(UW, self).__init__()
    
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([-0.5]*self.task_num, device=self.device))
    
    def backward(self, losses, args):
        loss = (losses/(2*self.loss_scale.exp())+self.loss_scale/2).sum()
        loss.backward()
        return (1/(2*torch.exp(self.loss_scale))).detach().cpu().numpy()