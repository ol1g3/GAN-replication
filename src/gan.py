from src.model import Model, Discriminator, Generator
from typing import Tuple
import torch

import torch.nn.functional as F

import torch.optim as optim
import numpy as np


class GAN(Model):
    def __init__(
        self,
        img_shape: Tuple,
        latent_dim: int,
        optim_g: optim,
        optim_d: optim,
        is_ragan: bool = False,
    ):
        super().__init__(img_shape, latent_dim, optim_d, optim_g)
        self.is_ragan = is_ragan

    def discriminate(self, data: torch.Tensor):
        return self.discriminator.forward(data)

    def generate(self, noise: torch.Tensor):
        return self.generator.forward(noise)

    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor):
        if self.is_ragan:
            adv_loss = torch.nn.BCEWithLogitsLoss()
            real_loss = adv_loss(
                real_pred - fake_pred.mean(0, keepdim=True),
                torch.ones_like(real_pred),
            )
            fake_loss = adv_loss(
                fake_pred - real_pred.mean(0, keepdim=True),
                torch.zeros_like(real_pred),
            )
        else:
            real_loss = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )

        total_loss = (real_loss + fake_loss) / 2
        return total_loss

    def generator_loss(self, fake_pred: torch.Tensor):
        return F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

    def train_discriminator(self, batch: torch.Tensor, noise: torch.Tensor) -> None:
        self.discriminator.zero_grad()
        fake_images = self.generate(noise)
        true_result = self.discriminate(batch)
        fake_result = self.discriminate(fake_images.detach())

        loss_d = self.discriminator_loss(true_result, fake_result)

        loss_d.backward()
        self.optimizer_d.step()

    def train_generator(self, noise: torch.Tensor) -> None:
        self.generator.zero_grad()
        fake_images = self.generate(noise)
        fake_result = self.discriminate(fake_images)

        loss_g = self.generator_loss(fake_result)

        loss_g.backward()
        self.optimizer_g.step()

    def train(self, dataset: np.array, epochs: int):

        for _ in range(epochs):
            for i in range(len(dataset)):
                batch = torch.from_numpy(dataset[i]).float()
                noise = torch.rand(batch.shape[0], self.latent_dim)
                self.train_discriminator(noise, batch)
                self.train_generator(noise)

        # somehow show the metrics after the epoch
        # somehow save the model
