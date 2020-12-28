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
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from citation_dataset import CitationDataset


def mask_and_pad(token_ids, token_type_ids, max_length, pad_token_id, label=1):
    """Get padded token ids, padded type ids, and attention mask."""
    pad_len = max_length - len(token_ids)
    attention_mask = [1] * len(token_ids) + [0] * pad_len
    padded_token_ids = token_ids + ([pad_token_id] * pad_len)
    padded_token_type_ids = token_type_ids + ([pad_token_id] * pad_len)
    return {
        "input_ids": torch.tensor(padded_token_ids), 
        "attention_mask": torch.tensor(attention_mask),
        "token_type_ids": torch.tensor(padded_token_type_ids),
        "labels": torch.tensor(label).float()
    }



def preprocess(tokenizer: BertTokenizer, x: Dict, max_length: int = 256) -> Dict:
    """Preprocess an example from the dataset.

    Example example:
    {
        "anchor": "A recent study examining post-mortem...",
        "positive": {
            "abstract": "Neurogranin (Ng) is a post-synaptic..."
            "pmid": "29700597"
        },
        "negative": [
            {
                "abstract": "Background Herpesviruses and bacteria..."
                "pmid": "32646510"
            },
            {
                "abstract": "Background Herpesviruses and bacteria..."
                "pmid": "32646510"
            },
            ...
        ]
    }
    """
    
    # Tokenize positive and negative examples
    pos = tokenizer.encode_plus(
        x["anchor"],
        x["positive"]["abstract"] or "",
        add_special_tokens=True,
        max_length=max_length,
        truncation=True
    )

    neg = tokenizer.encode_plus(
        x["anchor"],
        x["negative"]["abstract"] or "",
        add_special_tokens=True,
        max_length=max_length,
        truncation=True
    )

    return {
        "pos": mask_and_pad(pos['input_ids'],
                            pos['token_type_ids'],
                            max_length=max_length,
                            pad_token_id=tokenizer.pad_token_id,
                            label=1),
        "neg": mask_and_pad(neg['input_ids'],
                            neg['token_type_ids'],
                            max_length=max_length,
                            pad_token_id=tokenizer.pad_token_id,
                            label=0),
    }


class CitationNegModel(LightningModule):

    def __init__(self, hparams):
        super(CitationNegModel, self).__init__()

        self.hparams = hparams
        # num_labels=1 specifies a regression
        # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertConfig
        config = BertConfig(num_labels=1, return_dict=True)
        self.bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                        config=config)

    def forward(self, inputs):
        pos_loss = self.bert_model(**inputs['pos'])[0]
        neg_loss = self.bert_model(**inputs['neg'])[0]

        return torch.add(pos_loss, neg_loss)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        dataset = CitationDataset(partial(preprocess, tokenizer),
                                  page_size=self.hparams.batch_size*8)
        train_dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=18)
        return train_dataloader

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        return opt

if __name__=="__main__":
    args = {
        'batch_size': 12,
        'lr': 0.001,
        'b1': 0.5,
        'b2': 0.999,
    }
    model = CitationNegModel(Namespace(**args))

   # trainer = pl.Trainer(gpus=1, max_steps=50, max_epochs=2)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)
