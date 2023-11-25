import os, sys

from .unet_based import OxFordPetUnet

model_dict = {
    "unet" : {
        "oxford" : OxFordPetUnet
    }   
}