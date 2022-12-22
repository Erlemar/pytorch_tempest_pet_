import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from typing import List, Dict, Optional

import cv2
import imgaug.augmenters as iaa
import numpy as np
import numpy.typing as npt
import torch
from albumentations.core.composition import Compose
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch.utils.data import Dataset

from src.utils.ml_utils import iou


class DigitYOLODataset(Dataset):
    def __init__(
            self,
            bbox_annotations,
            image_names: List,
            transforms: Compose,
            labels: Optional[List[int]],
            anchors,
            img_path: str = '',
            mode: str = 'train',
            labels_to_ohe: bool = False,
            n_classes: int = 1,
            S=None
    ):
        """
        Object detection dataset.

        Args:
        """

        self.mode = mode
        self.transforms = transforms
        self.img_path = img_path
        self.bbox_annotations = bbox_annotations
        self.image_names = [i for i in image_names if i in bbox_annotations.keys()]
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

        if labels is not None:
            if not labels_to_ohe:
                self.labels = np.array(labels)
            else:
                self.labels = np.zeros((len(labels), n_classes))
                self.labels[np.arange(len(labels)), np.array(labels)] = 1

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        image = cv2.imread(self.image_names[idx], cv2.IMREAD_COLOR)
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        max_x = im.shape[0]
        max_y = im.shape[1]
        original_bboxes = self.bbox_annotations[self.image_names[idx]]
        if self.mode == 'train':
            bbi = []
            for b in [[b[0], b[1], b[0] + b[2], b[1] + b[3], b[4]] for b in original_bboxes]:
                bbi.append(BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=b[4]))

            bbs = BoundingBoxesOnImage(bbi, shape=im.shape)

            seq = iaa.Sequential([
                iaa.Affine(rotate=(-45, 45), scale=(0.9, 1.0),
                           translate_px={"x": (-20, 20), "y": (-20, 20)}, cval=255
                           )
            ])

            im, bbs_aug = seq(image=im, bounding_boxes=bbs)
            bboxes = [[b_.center_x / max_x, b_.center_y / max_y,
                       b_.width / max_x, b_.height / max_y, b_.label] for b_ in bbs_aug.bounding_boxes]
        elif self.mode == 'valid':

            bboxes = [[(box[0] + box[2] / 2) / max_x,
                       (box[1] + box[3] / 2) / max_y,
                       box[2] / max_x,
                       box[3] / max_y, box[4]] for box in original_bboxes]
        # print(self.transforms)
        augmentations = self.transforms(image=im, bboxes=bboxes)
        im = augmentations["image"]
        bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                # print(targets[0].shape, targets[1].shape, targets[2].shape)
                # print(idx, self.image_names[idx], scale_idx, anchor_on_scale, i, j)
                # print(box)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        #         image = self.transforms(image=im)['image']
        sample = {'image': im, 'target0': targets[0], 'target1': targets[1], 'target2': targets[2]}

        return sample

    def __len__(self) -> int:
        return len(self.image_names)
