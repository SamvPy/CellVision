import torch
import numpy as np
from lance.torch.data import SafeLanceDataset
from PIL import Image
from .preprocessing.augmentations import SimpleAugmentor, ImageToTensor

class ImageDataset(SafeLanceDataset):
    def __init__(self, uri, **kwargs):
        super().__init__(uri)

    def __getitem__(self, idx):
        # Calls __getitems__ from SafeLanceDatasets
        # This initializes workers in safe manner for batch fetching
        row = super().__getitem__(idx) 

        # Binnify the loaded sample
        return row
    
    def __getitems__(self, indices):
        # Initializes the workers and takes all batches in list
        batch = super().__getitems__(indices)

        # Binnify the spectrum and collect in dict (spectrum, trimer_vector)
        return [self.preprocess(sample) for sample in batch]
    
    def preprocess(self, sample):
        # Convert the lists in Images

        cell = Image.fromarray(
            np.array(sample['cell_diff_crop'], dtype=float)
        ).convert('L')

        # Create augmented views
        targets = ImageToTensor(cell).squeeze().unsqueeze(dim=0)
        aug_1 = SimpleAugmentor(cell).squeeze().unsqueeze(dim=0)
        aug_2 = SimpleAugmentor(cell).squeeze().unsqueeze(dim=0)
        
        return {
            'targets': targets, # add channel dimension
            'aug_1': aug_1,
            'aug_2': aug_2
        }