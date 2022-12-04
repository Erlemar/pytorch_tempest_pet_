from typing import Tuple

import torch
import torch.functional as F
from torch import nn

from src.utils.ml_utils import intersection_over_union


class VentilatorLoss(nn.Module):
    """
    Directly optimizes the competition metric (kaggle ventilator)
    """

    def __call__(self, preds, y, u_out):
        w = 1 - u_out
        mae = w * (y - preds).abs()
        mae = mae.sum(-1) / w.sum(-1)

        return mae


class MAE(nn.Module):
    def __call__(self, preds, y, u_out):
        # print(preds.shape, y.shape)
        return nn.L1Loss(preds, y).mean()


class DenseCrossEntropy(nn.Module):
    # Taken from: https://www.kaggle.com/pestipeti/plant-pathology-2020-pytorch
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        logprobs = F.log_softmax(logits, dim=-1)

        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()


class CutMixLoss:
    # https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/losses/cutmix.py
    def __init__(self, reduction: str = 'mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
        self, predictions: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float], train: bool = True
    ) -> torch.Tensor:
        if train:
            targets1, targets2, lam = targets
            loss = lam * self.criterion(predictions, targets1) + (1 - lam) * self.criterion(predictions, targets2)
        else:
            loss = self.criterion(predictions, targets)
        return loss


class MixupLoss:
    # https://github.com/hysts/pytorch_image_classification/blob/master/pytorch_image_classification/losses/mixup.py
    def __init__(self, reduction: str = 'mean'):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(
        self, predictions: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float], train: bool = True
    ) -> torch.Tensor:
        if train:
            targets1, targets2, lam = targets
            loss = lam * self.criterion(predictions, targets1) + (1 - lam) * self.criterion(predictions, targets2)
        else:
            loss = self.criterion(predictions, targets)
        return loss


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]),
            (target[..., 0:1][noobj]),
        )

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat(
            [self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1
        )
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        class_loss = self.entropy(
            (predictions[..., 5:][obj]),
            (target[..., 5][obj].long()),
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
