from torchvision import transforms
import torch

invnorm = transforms.Compose(
    [
        transforms.Normalize(mean = [0., 0., 0.], std = [1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize(mean = [-0.5, -0.5, -0.5], std = [1., 1., 1.]),
    ]
)

def invnorm255(x):
    x = x * 255

    return x.to(x.device, dtype=torch.uint8)