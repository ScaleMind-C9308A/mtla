import os, sys

from .depth import depth_error as __depth_error
from .seg import miou as __miou
from .seg import pixel_accuracy as __pixel_accuracy
from .acc import accuracy as __accuracy


metric_dict = {
    "depth" : __depth_error,
    "semantic_miou" : __miou,
    "semantic_pixel_accuracy" : __pixel_accuracy,
    "category_accuracy" : __accuracy,
    "attr_accuracy" : __accuracy
}

metric_batch = {
    "depth" : True,
    "semantic_miou" : True,
    "semantic_pixel_accuracy" : False,
    "category_accuracy" : False,
    "attr_accuracy" : False
}

# TODO: implement normal metrics