import os, sys

from .unet_based import OxFordPetUnet
from .segnet_based import OxFordPetSegNet

model_dict = {
    "unet" : {
        "oxford" : OxFordPetUnet
    },
    "segnet" : {
        "oxford" : OxFordPetSegNet
    }
}