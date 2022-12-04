from typing import Dict, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.utils.technical_utils import load_obj
from src.utils.ml_utils import cells_to_bboxes, non_max_suppression


class LitPetOD(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(LitPetOD, self).__init__()
        self.cfg = cfg
        self.model = load_obj(cfg.model.class_name)(cfg=cfg)
        # print(self.model)
        # self.register_buffer("scaled_anchors", torch.tensor([[[0.5600, 0.4400],
        #  [0.7600, 0.9600],
        #  [1.8000, 1.5600]],
        #
        # [[0.2800, 0.6000],
        #  [0.6000, 0.4400],
        #  [0.5600, 1.1600]],
        #
        # [[0.1600, 0.2400],
        #  [0.3200, 0.5600],
        #  [0.6400, 0.4800]]]))
        self.register_buffer("scaled_anchors", torch.tensor([[[1.6800, 1.3200],
         [2.2800, 2.8800],
         [5.4000, 4.6800]],

        [[0.8400, 1.8000],
         [1.8000, 1.3200],
         [1.6800, 3.4800]],

        [[0.4800, 0.7200],
         [0.9600, 1.6800],
         [1.9200, 1.4400]]]))
        self.register_buffer("anchors", torch.tensor([
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        ]))

        # self.register_buffer("S", torch.tensor([2, 4, 8]))
        self.register_buffer("S", torch.tensor([6, 12, 24]))

        if 'params' in self.cfg.loss:
            self.loss = load_obj(cfg.loss.class_name)(**self.cfg.loss.params)
        else:
            self.loss = load_obj(cfg.loss.class_name)()
        self.metrics = torch.nn.ModuleDict(
            {
                self.cfg.metric.metric.metric_name: load_obj(self.cfg.metric.metric.class_name)(
                    **cfg.metric.metric.params
                )
            }
        )
        if 'other_metrics' in self.cfg.metric.keys():
            for metric in self.cfg.metric.other_metrics:
                self.metrics.update({metric.metric_name: load_obj(metric.class_name)(**metric.params)})

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        if 'decoder_lr' in self.cfg.optimizer.params.keys():
            params = [
                {'params': self.model.decoder.parameters(), 'lr': self.cfg.optimizer.params.lr},
                {'params': self.model.encoder.parameters(), 'lr': self.cfg.optimizer.params.decoder_lr},
            ]
            optimizer = load_obj(self.cfg.optimizer.class_name)(params)

        else:
            optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return (
            [optimizer],
            [{'scheduler': scheduler, 'interval': self.cfg.scheduler.step, 'monitor': self.cfg.scheduler.monitor}],
        )

    def training_step(self, batch, *args, **kwargs):  # type: ignore
        x = batch['image']
        y0, y1, y2 = batch['target0'], batch['target1'], batch['target2']

        logits = self(x)
        # print('y0', y0.sum())
        # print('l', logits[0].sum())
        l0 = self.loss(logits[0], y0, self.scaled_anchors[0])
        l1 = self.loss(logits[1], y1, self.scaled_anchors[1])
        l2 = self.loss(logits[2], y2, self.scaled_anchors[2])
        loss = l0 + l1 + l2
        # print('train', l0, l1, l2, loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for metric in self.metrics:
            # score = self.metrics[metric](preds_, targets_)
            self.log(f'train_{metric}', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log(f'train_map_score', score['map'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log(f'train_map_50_score', score['map_50'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, *args, **kwargs):  # type: ignore
        x = batch['image']
        y0, y1, y2 = batch['target0'], batch['target1'], batch['target2']

        logits = self(x)
        # print('y0', y0.sum())
        # print('l', logits[0].sum())
        l0 = self.loss(logits[0], y0, self.scaled_anchors[0])
        l1 = self.loss(logits[1], y1, self.scaled_anchors[1])
        l2 = self.loss(logits[2], y2, self.scaled_anchors[2])
        loss = l0 + l1 + l2
        # print('valid', l0, l1, l2, loss)
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for metric in self.metrics:
            # score = self.metrics[metric](preds_, targets_)
            self.log(f'valid_{metric}', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log(f'valid_map_score', score['map'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log(f'valid_map_50_score', score['map_50'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

