from src.model import Model, Discriminator, Generator
import torch.nn as nn

from typing import Tuple

from torch.nn.utils import spectral_norm
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.models.vgg import vgg19

from src.gan import GAN


class SRGAN(Model, GAN):
    def __init__(self, img_shape: Tuple, latent_dim: int):
        super().__init__()
        self.initial = nn.Linear(latent_dim, 64 * 8 * 8)

        self.generator = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, 3, padding=1),
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        vgg = vgg19(pretrained=True).features
        self.vgg = vgg[:36]
        for param in self.vgg.parameters():
            param.requires_grad = False

    def generate(self, noise):
        x = self.initial(noise)
        x = x.view(-1, 64, 8, 8)
        ans = self.generator(x)

        return F.interpolate(
            ans, size=self.img_shape, mode="bilinear", align_corners=False
        )

    def content_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor):
        real_features = self.vgg(real_output)
        fake_features = self.vgg(fake_output)

        return nn.MSELoss()(fake_features, real_features)

    def adv_loss(self, fake_output: torch.Tensor):
        return -fake_output.mean()
