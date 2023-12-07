import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .core import Vanilla


class GradNorm(Vanilla):
    def __init__(self):
        super(GradNorm, self).__init__()
    
    def init_param(self):
        super().init_param()
        self.loss_scale = nn.Parameter(torch.tensor([1.0]*self.task_num, device=self.device))
    
    def backward(self, losses, args):
        alpha = args.alpha
        if args.epoch >= 1:
            loss_scale = args.task_num * F.softmax(self.loss_scale, dim=-1)
            grads = self._get_grads(losses, mode='backward')
                
            G_per_loss = torch.norm(loss_scale.unsqueeze(1)*grads, p=2, dim=-1)
            G = G_per_loss.mean(0)
            L_i = torch.Tensor([losses[tn].item()/self.train_loss_buffer[tn, 0] for tn in range(args.task_num)]).to(self.device)
            r_i = L_i/L_i.mean()
            constant_term = (G*(r_i**alpha)).detach()
            L_grad = (G_per_loss-constant_term).abs().sum(0)
            L_grad.backward()
            loss_weight = loss_scale.detach().clone()
            
            self._backward_new_grads(loss_weight, grads=grads)
            return loss_weight.cpu().numpy()
        else:
            loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
            loss.backward()
            return np.ones(self.task_num)