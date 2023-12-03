import os, sys
sys.path.append("/".join(__file__.split("/")[:-2]))

from .core import *
from abs import Core


class OxFordPetSegNet(Core):
    def __init__(self, args):
        super(OxFordPetSegNet, self).__init__(args)
        self.seg_n_classes = args.seg_n_classes
        self.cls_n_classes = args.cls_n_classes
        self.init_ch = args.init_ch

        self.encoder = nn.ModuleList(
            [
                DownConv2(3, self.init_ch, kernel_size=3),
                DownConv2(self.init_ch, self.init_ch*2, kernel_size=3),
                DownConv3(self.init_ch*2, self.init_ch*4, kernel_size=3),
                DownConv3(self.init_ch*4, self.init_ch*8, kernel_size=3),
            ]
        )

        self.decoder = nn.ModuleDict(
            {
                "cls" : nn.Sequential(
                    Down(self.init_ch*8, self.init_ch*16),
                    GlobalAvgPooling(),
                    nn.Linear(self.init_ch*16, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.cls_n_classes)
                ),
                "seg" : nn.ModuleList(
                    [
                        UpConv3(self.init_ch*8, self.init_ch*4, kernel_size=3),
                        UpConv3(self.init_ch*4, self.init_ch*2, kernel_size=3),
                        UpConv2(self.init_ch*2, self.init_ch, kernel_size=3),
                        UpConv2(self.init_ch, self.seg_n_classes, kernel_size=3)
                    ]
                )
            }
        )
    
    def forward(self, x):
        x, mp1_indices, shape1 = self.encoder[0](x)
        x, mp2_indices, shape2 = self.encoder[1](x)
        x, mp3_indices, shape3 = self.encoder[2](x)
        x, mp4_indices, shape4 = self.encoder[3](x)

        logits = self.decoder['cls'](x)

        x = self.decoder['seg'][0](x, mp4_indices, output_size=shape4)
        x = self.decoder['seg'][1](x, mp3_indices, output_size=shape3)
        x = self.decoder['seg'][2](x, mp2_indices, output_size=shape2)
        masks = self.decoder['seg'][3](x, mp1_indices, output_size=shape1)

        return {
            "category" : logits,
            "semantic" : masks
        }