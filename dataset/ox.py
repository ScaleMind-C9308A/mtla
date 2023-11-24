import os, sys
from typing import *
import cv2
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch.nn.functional as F
from torch import Tensor
import albumentations as A

_root = "/".join(__file__.split("/")[:-1]) + "/source/oxfordpet"

class CustomOxFordPet(Dataset):
    def __init__(self, root:str = _root, split = 'trainval'):
        self.root = root
        self._split = split
        self.__mode = "train" if self._split == 'trainval' else 'test'

        self.resize = A.Compose(
            [
                A.Resize(256, 256),
            ]
        )

        self.aug_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode = cv2.BORDER_CONSTANT),
            ]
        )

        self.norm = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self._images_folder = self.root + "/images"
        self._anns_folder = self.root + "/annotations"
        self._segs_folder = self._anns_folder + "/trimaps"

        print("Data Set Setting Up")
        image_ids = []
        self._labels = []
        with open(self._anns_folder + f"/{self._split}.txt") as file:
            for line in tqdm(file):
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in tqdm(
                    sorted(
                    {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                    key=lambda image_id_and_label: image_id_and_label[1],
                )
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder + f"/{image_id}.jpg" for image_id in tqdm(image_ids)]
        self._segs = [self._segs_folder + f"/{image_id}.png" for image_id in tqdm(image_ids)]
        print("Done")

    @staticmethod
    def process_mask(x):
        uniques = torch.unique(x, sorted = True)
        if uniques.shape[0] > 3:
            x[x == 0] = uniques[2]
            uniques = torch.unique(x, sorted = True)
        for i, v in enumerate(uniques):
            x[x == v] = i
        
        x = x.to(dtype=torch.long)
        onehot = F.one_hot(x.squeeze(1), 3).permute(0, 3, 1, 2)[0].float()
        return onehot

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self._images[idx]).convert("RGB"))
        mask = np.array(Image.open(self._segs[idx]))

        resized = self.resize(image = image, mask = mask)

        if self.__mode == 'train':
            transformed = self.aug_transforms(image = resized['image'], mask = resized['mask'])
            transformed_img = self.norm(image=transformed["image"])["image"]
            transformed_mask = transformed["mask"]
        else:
            transformed_img = self.norm(image=resized['image'])['image']
            transformed_mask = resized['mask']

        torch_img = torch.from_numpy(transformed_img).permute(-1, 0, 1).float()
        torch_mask = torch.from_numpy(transformed_mask).unsqueeze(-1).permute(-1, 0, 1).float()

        target = {
            "semantic" : self.process_mask(torch_mask),
            "category" : self._labels[idx],
        }

        return torch_img, target

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, m):
        if m not in ['train', 'test']:
            raise ValueError(f"mode cannot be {m} and must be ['train', 'test']")
        else:
            self.__mode = m