import glob
import os
from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.datasets.get_dataset import load_augs
from src.utils.technical_utils import load_obj


class DigitsDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        labels = []
        image_files_list = []
        digit_folders = [f'{self.cfg.datamodule.path}/{i}' for i in range(12)]
        for folder in digit_folders:
            for i, pic in enumerate(glob.glob(os.path.join(folder, '*.jpg'))):
                labels.append(int(folder.split('/')[-1]))
                image_files_list.append(pic)

        train_images, valid_images, train_labels, valid_labels = train_test_split(
            image_files_list, labels, stratify=labels, test_size=0.1
        )

        # train dataset
        dataset_class = load_obj(self.cfg.datamodule.class_name)

        # initialize augmentations
        train_augs = load_augs(self.cfg['augmentation']['train']['augs'])
        valid_augs = load_augs(self.cfg['augmentation']['valid']['augs'])

        self.train_dataset = dataset_class(
            image_names=train_images,
            labels=train_labels,
            transforms=train_augs,
            mode='train',
            labels_to_ohe=self.cfg.datamodule.labels_to_ohe,
            n_classes=self.cfg.training.n_classes,
        )
        self.valid_dataset = dataset_class(
            image_names=valid_images,
            labels=valid_labels,
            transforms=valid_augs,
            mode='valid',
            labels_to_ohe=self.cfg.datamodule.labels_to_ohe,
            n_classes=self.cfg.training.n_classes,
        )
        print('self.train_dataset', len(self.train_dataset))

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            multiprocessing_context=self.cfg.datamodule.multiprocessing_context,
            shuffle=True,
            collate_fn=load_obj(self.cfg.datamodule.collate_fn)(**self.cfg.datamodule.mix_params)
            if self.cfg.datamodule.collate_fn
            else None,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            multiprocessing_context=self.cfg.datamodule.multiprocessing_context,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )

        return valid_loader

    def test_dataloader(self):
        return None
