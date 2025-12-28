import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch

class ClampAndScale:
    def __init__(self, clamp_value=130):
        self.clamp_value = clamp_value

    def __call__(self, tensor):
        # tensor is in [0,1] after ToTensor()
        # convert clamp_value to normalized [0,1] range
        clamp_norm = self.clamp_value / 255.0
        
        # 1. clamp
        tensor = tensor.clamp(max=clamp_norm)
        
        # 2. scale to full [0,1]
        if clamp_norm > 0:
            tensor = tensor / clamp_norm

        return tensor
    

ImageToTensor = T.Compose(
    [
        T.ToTensor(),
        ClampAndScale(130)
    ]
)

SimpleAugmentor = T.Compose([
    # T.RandomRotation(degrees=180), # Allow to rotate the image any direction
    # T.RandomHorizontalFlip(p=0.5),
    # T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=0, translate=(0.4,0.4), scale=(1.0,1.0)),  # translations only, no scale
    T.ToTensor(),
    ClampAndScale(130)
])