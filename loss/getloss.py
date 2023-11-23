import os, sys
import torch
from torch import nn

from .dice_loss import dice_loss
from .gaussian_kls import gaussian_kls

# Loss Function Definition
__semantic_loss = nn.CrossEntropyLoss()

__binary_clf = nn.BCEWithLogitsLoss()

__cate_clf = nn.CrossEntropyLoss()

__recon_loss = nn.MSELoss(reduction='none')

def __recon_loss_fn(recon_x, x):
    return torch.mean(torch.sum(recon_loss(recon_x, x), dim=(1,2,3)))

loss_dict = {
    "seg" : __semantic_loss,
    "dce" : dice_loss,
    "atr" : __binary_clf,
    "clf" : __cate_clf,
    "rec" : __recon_loss_fn,
    "kld" : gaussian_kls
}