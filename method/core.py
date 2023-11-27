import os, sys
from typing import *

import torch
from torch import nn


class Vanilla(nn.Module):
    def __init__(self):
        super(Vanilla, self).__init__()
    
    def backward(self, losses: list = None, args = None):
        if args is None:
            raise ValueError('args cannot be None')
        total_loss = sum(losses)

        total_loss.backward()
    
    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad
    
    def _compute_grad(self, losses, mode):
        grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        for tn in range(self.task_num):
            if mode == 'backward':
                losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                grads[tn] = self._grad2vec()
            elif mode == 'autograd':
                grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                grads[tn] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError('No support {} mode for gradient computation')
            self.zero_grad_share_params()
        return grads
    
    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
        
    def _get_grads(self, losses, mode='backward'):
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode)
        return grads
    
    def _backward_new_grads(self, batch_weight, grads=None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            grads (torch.Tensor): The gradients of the shared parameters. 
        """
        # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
        new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
        self._reset_grad(new_grads)