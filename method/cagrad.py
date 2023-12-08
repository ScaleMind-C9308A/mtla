import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize

from .core import Vanilla


class CAGrad(AbsWeighting):
    def __init__(self):
        super(CAGrad, self).__init__()
        
    def backward(self, losses, args):
        calpha, rescale = args.calpha, args.rescale
        
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')
        
        GG = torch.matmul(grads, grads.t()).cpu() # [num_tasks, num_tasks]
        g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient

        x_start = np.ones(args.task_num) / self.task_num
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (calpha*g0_norm+1e-8).item()
        
        def objfn(x):
            return (x.reshape(1,-1).dot(A).dot(b.reshape(-1,1))+c * np.sqrt(x.reshape(1,-1).dot(A).dot(x.reshape(-1,1))+1e-8)).sum()
        
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(self.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = grads.mean(0) + lmbda * gw
        
        if rescale == 0:
            new_grads = g
        elif rescale == 1:
            new_grads = g / (1+calpha**2)
        elif rescale == 2:
            new_grads = g / (1 + calpha)
        else:
            raise ValueError('No support rescale type {}'.format(rescale))
        
        self._reset_grad(new_grads)
        return w_cpu