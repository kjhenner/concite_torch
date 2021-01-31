import json
import time
from argparse import Namespace
from functools import partial
from typing import Dict

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from transformers import BertConfig, BertForNextSentencePrediction, BertTokenizerFast, AdamW

from datasets.jsonl_citation_dataset import JsonlCitationDataset
from datasets.json_citation_dataset import JsonCitationDataset
from datasets.dict_dataset import DictDataset
from utils.utils import msearch_contexts, collate_hits
from metrics.ndcg import nDCG

from elasticsearch import Elasticsearch


def preprocess(tokenizer, is_labeled, ex: Dict, max_length: int = 256) -> Dict:
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
        text=ex["anchor"],
        text_pair=ex["example"]["abstract"] or "",
        add_special_tokens=True,
        max_length=max_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    if is_labeled:
        tokenized['labels'] = ex['label']
    tokenized['source'] = {
        'pmid': ex['example']['pmid'],
        'abstract': ex['example']['abstract'] or '',
        'context': ex['anchor']
    }

    return tokenized


def rerank_preprocess(tokenizer,
                      is_labeled: bool,
                      es_config: Dict,
                      ex: Dict,
                      max_length: int = 256) -> Dict:
    """Preprocess an example from the dataset for the reranking task.

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
    es_client = Elasticsearch(**es_config)
    tokenized = tokenizer(
        text=ex["anchor"],
        text_pair=ex["example"]["abstract"] or "",
        add_special_tokens=True,
        max_length=max_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    if is_labeled:
        tokenized['labels'] = ex['label']
    tokenized['source'] = {
        'pmid': ex['example']['pmid'],
        'abstract': ex['example']['abstract'] or '',
        'context': ex['anchor']
    }
    responses = msearch_contexts(es_client,
                                 source['context'],
                                 size=self.hparams.es_search_size,
                                 fields=self.hparams.es_search_fields)['responses']
    for i, response in enumerate(responses):
        cited_pmid = source['pmid'][i]
        hit_examples = []
        #print(f"Number of hits: {len(response['hits']['hits'])}")
        for hit in response['hits']['hits']:
            hit_examples.append({
                "anchor": source['context'][i],
                "example": {
                    "abstract": hit["_source"]["abstract"],
                    "pmid": hit["_source"]["pmid"],
                }
            })
        if not hit_examples:
            continue
        #print(f"Number of hit examples: {len(hit_examples)}")
        #print(f"hit_examples[0]: {hit_examples[0]}")
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
        self.val_ndcg = nDCG()

        bert_config = BertConfig.from_pretrained("bert-base-uncased",
                                                 hidden_dropout_config=self.hparams.dropout,
                                                 attention_probs_dropout_prob=self.hparams.dropout,
                                                 num_labels=2)
        self.transformer = BertForNextSentencePrediction.from_pretrained("bert-base-uncased", config=bert_config)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)

        self.es_config = {
            'hosts': ['localhost'],
            'scheme': 'http',
            'verify_certs': False,
            'port': 9200
        }
        # Don't create a client until validation epoch start to avoid
        # pickling problems.
        self.es_client = None

    def forward(self, inputs):
        for k, v in inputs.items():
            inputs[k] = v.squeeze()
        return self.transformer(**inputs)

    def training_step(self, batch, batch_idx):
        batch.pop('source')
        output = self(batch)
        loss = output.loss
        preds = output.logits.max(1).indices
        self.log('acc', self.acc(preds, batch['labels']))
        return loss

    def training_epoch_end(self, outs):
        self.log('acc', self.val_acc.compute())

    def validation_step(self, batch, batch_idx):
        self.es_client = self.es_client or Elasticsearch(**self.es_config)
        source = batch.pop('source')
        #ts = time.time()
        responses = msearch_contexts(self.es_client,
                                     source['context'],
                                     size=self.hparams.es_search_size,
                                     fields=self.hparams.es_search_fields)['responses']
        #print(f"validation msearch_contexts took {time.time() - ts:.2f} s")
        #print(f"Number of responses: {len(responses)}")
        # For each example in the batch, we get the top k results from Elasticsearch
        outer_ts = time.time()
        for i, response in enumerate(responses):
            cited_pmid = source['pmid'][i]
            hit_examples = []
            #print(f"Number of hits: {len(response['hits']['hits'])}")
            for hit in response['hits']['hits']:
                hit_examples.append({
                    "anchor": source['context'][i],
                    "example": {
                        "abstract": hit["_source"]["abstract"],
                        "pmid": hit["_source"]["pmid"],
                    }
                })
            if not hit_examples:
                continue
            #print(f"Number of hit examples: {len(hit_examples)}")
            #print(f"hit_examples[0]: {hit_examples[0]}")
            rerank_dataset = DictDataset(hit_examples,
                                         partial(preprocess, self.tokenizer, False))
            rerank_dataloader = DataLoader(rerank_dataset,
                                           batch_size=32,
                                           num_workers=self.hparams.dataloader_workers)
            output = []
            #inner_ts = time.time()
            for rerank_batch in rerank_dataloader:
                rerank_batch.pop('source')
                batch_output = self(self.transfer_batch_to_device(rerank_batch))
                output.append(batch_output.logits)
            #print(f"forward on rerank examples took {time.time() - inner_ts:.2f} s")

            y_pred = torch.cat(output)[:,1].cpu()

            # Padded ground-truth label vector
            y_true = [float(hit["example"].get('pmid') == cited_pmid) for hit in hit_examples]
            y_true += [0] * (self.hparams.es_search_size - len(y_true))

            self.log('val_nDCG', self.val_ndcg(y_true, y_pred))

        print(f"responses loop took {time.time() - outer_ts:.2f} s")
        output = self(batch)
        loss = output.loss
        self.log('val_loss', loss)
        preds = output.logits.max(1).indices
        self.log('val_acc', self.val_acc(preds, batch['labels']))
        return loss

    def train_dataloader(self):
        dataset = JsonlCitationDataset(self.hparams.train_data, partial(preprocess, self.tokenizer, True))
        train_dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.dataloader_workers)
        return train_dataloader

    def val_dataloader(self):
        dataset = JsonlCitationDataset(self.hparams.val_data, partial(preprocess, self.tokenizer, True))
        val_dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.dataloader_workers)
        return val_dataloader

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
        'val_data': '/mnt/atlas/cit_pred_jsonlines/validate.jsonl',
        'dataloader_workers': 1,
        'es_search_fields': ['abstract^2', 'context^3', 'text'],
        'es_search_size': 50
    }
    model = CitationNegModel(Namespace(**args))
    trainer = pl.Trainer(gpus=1, max_epochs=1)
    #trainer = pl.Trainer(gpus=2, max_epochs=1, distributed_backend='ddp')
    trainer.fit(model)
    torch.save(model.state_dict(), '/mnt/atlas/models/model.pt')
