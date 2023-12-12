import os, sys
import torch
from torch import nn

from .dice_loss import dice_loss as __dice_loss
from .gaussian_kld import gaussian_kls as __gaussian_kls

# Loss Function Definition
__semantic_loss = nn.CrossEntropyLoss()

__binary_clf = nn.BCEWithLogitsLoss()

__cate_clf = nn.CrossEntropyLoss()

__recon_loss = nn.MSELoss(reduction='none')

__depth_loss = nn.MSELoss()

def __recon_loss_fn(recon_x, x):
    return torch.mean(torch.sum(__recon_loss(recon_x, x), dim=(1,2,3)))

loss_dict = {
    "semantic" : __semantic_loss,
    "semantic_dice" : __dice_loss,
    "attr" : __binary_clf,
    "category" : __cate_clf,
    "reconstruction" : __recon_loss_fn,
    "kld" : __gaussian_kls,
    "depth" : __depth_loss


    # TODO: implement normal loss function and deep metric loss function
}

loss_batch = {
    "semantic" : True,
    "semantic_dice" : True,
    "attr" : True,
    "category" : True,
    "reconstruction" : True,
    "kld" : True,
    "depth" : True
}