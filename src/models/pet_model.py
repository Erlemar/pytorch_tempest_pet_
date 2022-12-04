from omegaconf import DictConfig

import torch.nn as nn
import torch.nn.functional as F
from src.utils.technical_utils import load_obj


class Net(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, cfg.training.n_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        bs = x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(bs, -1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits
