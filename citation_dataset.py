from torch.utils.data import IterableDataset
import torch
import tqdm
import numpy as np
import json
import random
from pathlib import Path
from elasticsearch import Elasticsearch
from typing import List, Text

import matplotlib.pyplot as plt
import re

class CitationDataset(IterableDataset):

    def __init__(self,
                 preprocess,
                 es_hosts=['localhost'],
                 es_port='9200',
                 page_size: int = 32):
        super().__init__()

        self.preprocess = preprocess
        self.es_hosts = es_hosts
        self.es_port = es_port
        self.page_size = page_size

        self.client = Elasticsearch(
            hosts=es_hosts,
            scheme='http',
            verify_certs=False,
            port=es_port
        )

    def get_negative_batch(self):
        body = {
            "size": self.page_size,
		    "query": {
                "function_score": {
                    "random_score": {}
                }
            },
            "_source": ['abstract', 'pmid']
        }
        return [hit['_source'] for hit in self.client.search(index='pubmed_articles',
                                                             body=body)['hits']['hits']]

    def get_positive_batch(self, pmids: List[Text]):
        body = {
            "ids": pmids
        }

        return [hit["_source"] for hit in self.client.mget(index='pubmed_articles',
                                                           _source=['abstract', 'pmid'],
                                                           body=body)['docs']]

    def __iter__(self):
        body = {
            "query": {
                "bool": {
                    "must": {"match": {"internal": True}},
                    "filter": {"range": {"year": {"gt": 2017}}}
                }
            }
        }
        data = self.client.search(index='pubmed_edges',
                                  scroll='10m',
                                  size=self.page_size,
                                  body=body)
        sid = data['_scroll_id']
        scroll_size = len(data['hits']['hits'])
        while scroll_size:
            neg_batch = self.get_negative_batch()
            pos_batch = self.get_positive_batch([hit['_source']['cited_pmid'] for hit in data['hits']['hits']])
            for anchor, pos, neg in zip(data['hits']['hits'], pos_batch, neg_batch):
                yield self.preprocess({
                    'anchor': anchor['_source']['context'],
                    'positive': pos,
                    'negative': neg
                })
            data = self.client.scroll(scroll_id=sid, scroll='10m')
            sid = data['_scroll_id']
            scroll_size = len(data['hits']['hits'])

    def __repr__(self):
        return f'Dataset class with for pubmed citation edges'

if __name__ == "__main__":
    test = CitationDataset(lambda x:x)
    for item in test:
        print(item)
