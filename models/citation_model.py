import json
from argparse import Namespace
from functools import partial
from typing import Dict

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from transformers import BertConfig, BertForNextSentencePrediction, BertTokenizerFast, AdamW

from jsonl_citation_dataset import JsonlCitationDataset


def preprocess(tokenizer, x: Dict, max_length: int = 256) -> Dict:
    """Preprocess an example from the dataset.

    Example example:
    {
        "anchor": "A recent study examining post-mortem...",
        "example": {
            "abstract": "Neurogranin (Ng) is a post-synaptic...",
            "pmid": "29700597"
        },
        "label": 1
        }
    }
    """

    tokenized = tokenizer(
        text=x["anchor"],
        text_pair=x["example"]["abstract"] or "",
        add_special_tokens=True,
        max_length=max_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    tokenized['labels'] = x['label']
    return tokenized


class CitationNegModel(LightningModule):

    def __init__(self, hparams):
        super(CitationNegModel, self).__init__()

        self.hparams = hparams

#        self.train_metrics = nn.ModuleDict({
#            'Acc': pl.metrics.Accuracy(),
#            'F1': pl.metrics.classification.F1(num_classes=2),
#            'P': pl.metrics.classification.Precision(num_classes=2),
#            'R': pl.metrics.classification.Recall(num_classes=2)
#        })

        self.acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

        bert_config = BertConfig.from_pretrained("bert-base-uncased",
                                                 hidden_dropout_config=self.hparams.dropout,
                                                 attention_probs_dropout_prob=self.hparams.dropout,
                                                 num_labels=2)
        self.transformer = BertForNextSentencePrediction.from_pretrained("bert-base-uncased", config=bert_config)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)

    def forward(self, inputs):
        for k, v in inputs.items():
            inputs[k] = v.squeeze()
        return self.transformer(**inputs)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        # self.log('loss', loss, prog_bar=True)
        preds = output.logits.max(1).indices
        self.log('acc', self.acc(preds, batch['labels']))
        return loss

    def training_epoch_end(self, outs):
        self.log('acc', self.val_acc.compute())

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log('val_loss', loss)
        preds = output.logits.max(1).indices
        self.log('val_acc', self.val_acc(preds, batch['labels']))

    def train_dataloader(self):
        dataset = JsonlCitationDataset(self.hparams.train_data, partial(preprocess, self.tokenizer))
        train_dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=10)
        return train_dataloader

    def val_dataloader(self):
        dataset = JsonlCitationDataset(self.hparams.val_data, partial(preprocess, self.tokenizer))
        train_dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=10)
        return train_dataloader

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        # Freeze transformer layers
        if self.hparams.freeze_layers:
            for n, p in self.transformer.bert.encoder.layer.named_parameters():
                if int(n.split('.')[0]) in self.hparams.freeze_layers:
                    p.requires_grad = False

        print(self)
        print(json.dumps(self.hparams, indent=2))
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0}
        ]
        opt = AdamW(optimizer_grouped_parameters, lr=lr, betas=(b1, b2))
        return opt


if __name__ == "__main__":
    args = {
        'batch_size': 16,
        'lr': 0.0001,
        'b1': 0.9,
        'b2': 0.999,
        'freeze_layers': list(range(11)),
        'dropout': 0.4,
        'weight_decay': 0.15,
        'train_data': '/mnt/atlas/cit_pred_jsonlines/train.jsonl',
        'val_data': '/mnt/atlas/cit_pred_jsonlines/validate.jsonl'
    }
    model = CitationNegModel(Namespace(**args))
    trainer = pl.Trainer(gpus=2, max_epochs=1, accelerator='ddp')
    trainer.fit(model)
    torch.save(model.state_dict(), '/mnt/atlas/models/model.pt')
