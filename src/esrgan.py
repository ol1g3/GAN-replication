from src.model import Model, Discriminator, Generator
import torch.nn as nn

from torch.nn.utils import spectral_norm
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.models.vgg import vgg19
from src.gan import GAN

from typing import Tuple


class ESRGAN(Model, GAN):  # TODO
    def __init__(self, img_shape: Tuple, latent_dim: int):
        super().__init__(img_shape, latent_dim)

        self.initial = nn.Linear(latent_dim, 64 * 8 * 8)
        self.generator = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 3, 3, padding=1),
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, padding=1),
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

    def content_loss(self, fake_output, real_output):
        fake_vgg = self.vgg(fake_output)
        real_vgg = self.vgg(real_output)

        return nn.MSELoss()(fake_vgg, real_vgg)
