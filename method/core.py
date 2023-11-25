import os, sys
from typing import *

import torch
from torch import nn


class Vanilla(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Vanilla, self).__init__()
    
    def backward(self, model: nn.Module = None, losses: list = None):
        total_loss = sum(losses)

        total_loss.backward()