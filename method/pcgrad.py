import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .core import Vanilla

class PCGrad(Vanilla):
    def __init__(self):
        super(PCGrad, self).__init__()
        
    def backward(self, losses, args):
        batch_weight = np.ones(len(losses))
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward') # [task_num, grad_dim]
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            random.shuffle(task_index)
            for tn_j in task_index:
                g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                if g_ij < 0:
                    pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2)+1e-8)
                    batch_weight[tn_j] -= (g_ij/(grads[tn_j].norm().pow(2)+1e-8)).item()
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight