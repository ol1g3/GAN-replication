import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import numpy as np

from typing import Tuple

from torch.nn.utils import spectral_norm
from torchvision.models.vgg import vgg19


class Model:
    def __init__(
        self,
        img_shape: Tuple,
        latent_dim: int,
        optim_g: optim,
        optim_d: optim,
    ):
        self.optim_g = optim_g
        self.optim_d = optim_d

        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.generator = Generator()
        self.discriminator = Discriminator()


class Discriminator(nn.Module, Model):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        real_or_fake = self.model(x)
        return real_or_fake


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(self.img_shape))),
            nn.Sigmoid(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
