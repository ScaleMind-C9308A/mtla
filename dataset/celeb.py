import os, sys
from torchvision import transforms
from torchvision.datasets import CelebA


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor()
    ]
)


class CustomCeleb(CelebA):
    def __init__(self, root="~/data", split='train', download=True, transform=transform):
        super().__init__(root=root_dir, split=split, download=download, transform=transform, target_type = ['attr', 'identity'])
        self.attr_names = self.attr_names[:-1]

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        attr = target[0]
        identity = target[1]

        target = {
            "reconstruction" : img,
            "identity" : identity
        }

        attr_dict = {
            f"attr_{self.attr_names[idx]}" : attr[idx] for idx in range(attr.shape[0])
        }

        target.update(attr_dict)

        return img, target