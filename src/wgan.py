from src.model import Model, Discriminator, Generator
import torch.nn as nn

from typing import Tuple

from torch.nn.utils import spectral_norm
import numpy as np
import torch
import torch.nn.functional as F

from src.gan import GAN

pen = 0.001


class WGAN(Model, GAN):
    def __init__(self, img_shape: Tuple, latent_dim: int):
        super().__init__(img_shape, latent_dim)

        self.generator = Generator()
        self.generator.model = nn.Sequential(
            *list(self.generator.model.children())[:-1]
        )

        self.discriminator = Discriminator()
        self.discriminator.model = nn.Sequential(
            spectral_norm(nn.Linear(int(np.prod(self.img_shape)), 512)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor):
        alpha = 0.2

        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        d_interpolates = self.discriminate(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(real_samples.shape[0], -1)
        gradients_norm = gradients.norm(2, dim=1)
        return ((gradients_norm - 1) ** 2).mean()

    def train_discriminator(self, batch: torch.Tensor, noise: torch.Tensor) -> None:
        self.discriminator.zero_grad()
        fake_images = self.generate(noise)
        true_result = self.discriminate(batch)
        fake_result = self.discriminate(fake_images.detach())

        gp = self.gradient_penalty(self.discriminator, batch, fake_images.detach())
        loss_d = -torch.mean(true_result) + torch.mean(fake_result) + pen * gp

        loss_d.backward()
        self.optimizer_d.step()

    def train(self, dataset: np.array, epochs: int, critic_trainings: int = 3):
        for _ in range(epochs):
            for i in range(len(dataset)):
                batch = torch.from_numpy(dataset[i]).float()

                for __ in range(critic_trainings):
                    noise = torch.rand(batch.shape[0], self.latent_dim)
                    self.train_discriminator(noise, batch)

                self.train_generator(noise)
