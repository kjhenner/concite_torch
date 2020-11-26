import os
import argparse
from argparse import Namespace, ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pubmed_dataset import PubmedDataset

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

class Discriminator(LightningModule):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape

        def conv_block(in_channels, out_channels, normalize=True, kernel_size=16, stride=2, padding=0):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
                *conv_block(1, 256),
                *conv_block(256, 256),
                *conv_block(256, 256),
                *conv_block(256, 1),
        )

        self.adv_layer = nn.Sequential(nn.Linear(324, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img.view(img.shape[0], *self.img_shape))
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)

class GAN(LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        img_shape = (1, 512, 512)
        self.generator = Generator(latent_dim=hparams.latent_dim,
                                   img_shape=img_shape,
                                   batch_size=hparams.batch_size)
        self.discriminator = Discriminator(img_shape=img_shape)

        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_idx == 0:
            # sample noise
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim)

            if self.on_gpu:
                z = z.cuda(imgs.device.index)

            # generate images
            self.generated_imgs = self(z)

            valid = torch.ones(imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:

            valid = torch.ones(imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            fake = torch.zeros(imgs.size(0), 1)
            if self.on_gpu:
                fake = fake.cuda(imgs.device.index)

            fake_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        dataset = ImageDataset('../data/heightmaps', '.*h\.png')
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def on_epoch_end(self):
        z = torch.randn(self.last_imgs.shape[0], self.hparams.latent_dim)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

if __name__=="__main__":
    args = {
        'batch_size': 16,
        'lr': 0.001,
        'b1': 0.5,
        'b2': 0.999,
        'latent_dim': 100
    }
    gan_model = GAN(Namespace(**args))

    trainer = pl.Trainer(gpus=2)
    trainer.fit(gan_model)
