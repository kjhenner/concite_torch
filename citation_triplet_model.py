import os
import argparse
from argparse import Namespace, ArgumentParser
from collections import OrderedDict
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

from transformers import BertTokenizer

from citation_dataset import CitationDataset


def preprocess(tokenizer: BertTokenizer, x: Dict, max_length: int = 256) -> Dict:
    """Preprocess an example from the dataset.

    Example example:
    {
        "anchor": "A recent study examining post-mortem...",
        "positive": {
            "abstract": "Neurogranin (Ng) is a post-synaptic..."
            "pmid": "29700597"
        },
        "negative": {
            "abstract": "Background Herpesviruses and bacteria..."
            "pmid": "32646510"
        }
    }
    """
    
    # Tokenize positive and negative examples
    pos = tokenizer.encode_plus(
        x["anchor"],
        x["positive"]["abstract"] or "",
        add_special_tokens=True,
        max_length=max_length
    )

    neg = tokenizer.encode_plus(
        x["anchor"],
        x["positive"]["abstract"] or "",
        add_special_tokens=True,
        max_length=max_length
    )

    # Handle attention mask and padding
    pos_pad_len = max_length - len(pos["input_ids"])
    neg_pad_len = max_length - len(neg["input_ids"])

    pos_attention_mask = [1] * len(pos["input_ids"]) + [0] * pos_pad_len
    neg_attention_mask = [1] * len(neg["input_ids"]) + [0] * neg_pad_len

    pad_token_id = tokenizer.pad_token_id

    pos["input_ids"] = pos["input_ids"] + (pad_token_id * pos_pad_len)
    neg["input_ids"] = neg["input_ids"] + (pad_token_id * neg_pad_len)

    pos["token_type_ids"] = pos["token_type_ids"] + (pad_token_id * pos_pad_len)
    neg["token_type_ids"] = neg["token_type_ids"] + (pad_token_id * neg_pad_len)

    return {
        "pos": {
            "input_ids": torch.tensor(pos["input_ids"]),
            "attention_mask": torch.tensor(pos_attention_mask),
            "token_type_ids": torch.tensor(pos["token_type_ids"])
        },
        "neg": {
            "input_ids": torch.tensor(neg["input_ids"]),
            "attention_mask": torch.tensor(neg_attention_mask),
            "token_type_ids": torch.tensor(neg["token_type_ids"])
        }
    }


class CitationTripletModel(LightningModule):

    def __init__(self, hparams):
        super(CitationTripletModel, self).__init__()

        self.hparams = hparams

        self.model = nn.Sequential(
            nn.Linear(768, 200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 2),
        )

        #self.node_embedder = node
        self.loss_function = nn.MSELoss()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs = batch
        self.last_imgs = imgs

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

    def train_dataloader(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        dataset = CitationDataset(partial(preprocess, tokenizer), page_size=self.hparams.batch_size)
        train_dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size)
        return train_dataloader

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2))
        return opt

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
    model = CitationTripletModel(Namespace(**args))

    trainer = pl.Trainer(gpus=2)
    trainer.fit(model)
