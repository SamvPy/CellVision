import torchvision.transforms as T
import matplotlib.pyplot as plt

ImageToTensor = T.Compose(
    [T.ToTensor()]
)

SimpleAugmentor = T.Compose([
    T.RandomRotation(degrees=180), # Allow to rotate the image any direction
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=0, translate=(0.2,0.2), scale=(1.0,1.0)),  # translations only, no scale
    T.ToTensor(),
])