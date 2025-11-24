"""
Stores all data preparation steps in the DataModule class of Pytorch Lightning.

Two data modules are defined here which differ in how data is parsed for storage:
- nine_species benchmark dataset
- Reprocessing: Pairs of MGF/MzML and PSMLists

Data preparation relies on DepthCharge.
"""

import lightning as L
import lance
import os
import logging
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(filename="datamodules.log", level=logging.INFO)

class ImageDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "path/to/dir",
            batch_size: int = 32,
            name_dataset_val: str = '',
            name_dataset_test: str = '',
            name_dataset_train: str = '',
            name_dataset_predict: str = '',
            num_workers: int = 12,
            dataset_cls: Dataset = Dataset,
            dataset_kwargs: dict = {},
            **kwargs
        ):
        super().__init__()
        self.data_dir = data_dir
        self.name_dataset_val = name_dataset_val
        self.name_dataset_test = name_dataset_test
        self.name_dataset_train = name_dataset_train
        self.name_dataset_predict = name_dataset_predict

        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.ds_train, self.ds_val, self.ds_test, self.ds_predict = None, None, None, None

    #TODO: Implement the prepare data function for reproducible and transparent
    # description of data preparation

    def setup(self, stage: str):
        def _path(dataset_name):
            return os.path.join(self.data_dir, dataset_name)

        if stage in (None, "fit"):
            if os.path.exists(_path(self.name_dataset_train)):
                self.ds_train = self.dataset_cls(uri=_path(self.name_dataset_train), **self.dataset_kwargs)
            else:
                raise Exception(f'Path not found to train dir: {_path(self.name_dataset_train)}')
            if os.path.exists(_path(self.name_dataset_val)):
                self.ds_val = self.dataset_cls(uri=_path(self.name_dataset_val), **self.dataset_kwargs)
            else:
                raise Exception(f'Path not found to val dir: {_path(self.name_dataset_val)}')
        
        if stage in (None, "test"):
            if os.path.exists(_path(self.name_dataset_test)):
                self.ds_test = self.dataset_cls(uri=_path(self.name_dataset_test), **self.dataset_kwargs)
            else:
                raise Exception(f'Path not found to test dir: {_path(self.name_dataset_test)}')

        if stage in (None, "predict"):
            if os.path.exists(_path(self.name_dataset_predict)):
                self.ds_predict = self.dataset_cls(uri=_path(self.name_dataset_predict), **self.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            persistent_workers=True,
            multiprocessing_context='spawn'
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            multiprocessing_context='spawn'
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ds_predict,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            multiprocessing_context='spawn'
        )