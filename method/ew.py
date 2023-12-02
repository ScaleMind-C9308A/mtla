import os, sys
from typing import *

import torch
from torch import nn
from .core import Vanilla 


class EW(Vanilla):
    def __init__(self):
        super(EW, self).__init__()
    
    def backward(self, losses: list = None, args = None):
        if args is None:
            raise ValueError('args cannot be None')
        total_loss = sum(losses) / len(losses)

        total_loss.backward()