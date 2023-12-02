import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .core import Vanilla


class RLW(Vanilla):
    def __init__(self):
        super(RLW, self).__init__()
        
    def backward(self, losses, args):
        batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()