import glob
import os
from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.datasets.get_dataset import load_augs
from src.utils.technical_utils import load_obj
import json

class DigitsYoloDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        labels = []
        image_files_list = []
        digit_folders = [f'{self.cfg.datamodule.path}/{i}' for i in range(12)] + [f'{self.cfg.datamodule.path}/multiple']
        # multiple check if is in images
        for folder in digit_folders:
            for i, pic in enumerate(glob.glob(os.path.join(folder, '*.jpg'))):
                if folder.split('/')[-1] == 'multiple':
                    labels.append(13)
                    # image_files_list.append(pic)
                    # labels.append(13)
                    # image_files_list.append(pic)
                    # labels.append(13)
                else:
                    labels.append(int(folder.split('/')[-1]))
                image_files_list.append(pic)

        train_images, valid_images, train_labels, valid_labels = train_test_split(image_files_list,
                                                                                  labels,
                                                                                  stratify=labels,
                                                                                  test_size=0.1)
        # _, train_images, _, train_labels = train_test_split(train_images,
        #                                                                           train_labels,
        #                                                                           stratify=train_labels,
        #                                                                           test_size=0.1)
        # train_images = train_images[:128] * 5
        # valid_images = valid_images[:128] * 5
        # train_labels = train_labels[:128] * 3
        # valid_labels = valid_labels[:128] * 3
        with open(f'{self.cfg.datamodule.path}/bbox_annotations2.json', 'r') as f:
            bbox_annotations = json.load(f)

        # train dataset
        dataset_class = load_obj(self.cfg.datamodule.class_name)

        # initialize augmentations
        train_augs = load_augs(self.cfg['augmentation']['train']['augs'])
        valid_augs = load_augs(self.cfg['augmentation']['valid']['augs'])
        print('train_augs', train_augs)
        print('valid_augs', valid_augs)

        anchors = [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        ]

        # S = [2, 4, 8]
        S = [6, 12, 24]

        self.train_dataset = dataset_class(
            bbox_annotations=bbox_annotations,
            image_names=train_images,
            labels=train_labels,
            transforms=train_augs,
            mode='train',
            labels_to_ohe=self.cfg.datamodule.labels_to_ohe,
            n_classes=self.cfg.training.n_classes,
            S=S,
            anchors=anchors
        )
        print('FFFFFF', len(self.train_dataset))
        self.valid_dataset = dataset_class(
            bbox_annotations=bbox_annotations,
            image_names=valid_images,
            labels=valid_labels,
            transforms=valid_augs,
            mode='valid',
            labels_to_ohe=self.cfg.datamodule.labels_to_ohe,
            n_classes=self.cfg.training.n_classes,
            S=S,
            anchors=anchors
        )
        # print('self.train_dataset', len(self.train_dataset))

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
