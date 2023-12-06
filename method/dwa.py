import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .core import Vanilla


class DWA(Vanilla):
    def __init__(self):
        super(DWA, self).__init__()
    
    def backward(self, losses, args):
        if args.epoch > 1:
            w_i = torch.Tensor(
                self.train_loss_buffer[:, args.epoch-1] / self.train_loss_buffer[:, args.epoch-2]
                ).to(self.device)
            batch_weight = args.task_num * F.softmax(w_i/args.T, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(self.device)
        
        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()