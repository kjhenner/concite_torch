from torch.utils.data import IterableDataset
import os
import tqdm
import json
import random
from elasticsearch import Elasticsearch
from typing import List, Text


class CitationDataset(IterableDataset):

    def __init__(self,
                 preprocess,
                 es_hosts=['localhost'],
                 es_port='9200',
                 page_size: int = 32,
                 negative_multiplier: int = 3):
        super().__init__()

        self.preprocess = preprocess
        self.es_hosts = es_hosts
        self.es_port = es_port
        self.page_size = page_size
        self.negative_multiplier = negative_multiplier

        self.client = Elasticsearch(
            hosts=es_hosts,
            scheme='http',
            verify_certs=False,
            port=es_port
        )

        self.total_articles = self.get_total_articles()
        self.total_examples = self.total_articles + self.total_articles * negative_multiplier

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

    def get_total_articles(self):
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
        return data['hits']['total']['value']

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
            examples = []
            pos_batch = self.get_positive_batch([hit['_source']['cited_pmid'] for hit in data['hits']['hits']])
            for anchor, pos in zip(data['hits']['hits'], pos_batch):
                examples.append(self.preprocess({
                    'anchor': anchor['_source']['context'],
                    'example': pos,
                    'label': 1,
                }))
            for _ in range(self.negative_multiplier):
                neg_batch = self.get_negative_batch()
                for anchor, neg in zip(data['hits']['hits'], neg_batch):
                    examples.append(self.preprocess({
                        'anchor': anchor['_source']['context'],
                        'example': neg,
                        'label': 0,
                    }))
            random.shuffle(examples)
            for example in examples:
                yield example
            data = self.client.scroll(scroll_id=sid, scroll='10m')
            sid = data['_scroll_id']
            scroll_size = len(data['hits']['hits'])

    def __repr__(self):
        return 'Dataset class with for pubmed citation edges'

    def to_jsonlines(self, out_dir: Text, test_prop: int = 0.1, val_prop: int = 0.1):
        progress = tqdm.tqdm(unit="example", total=self.total_examples)
        with open(os.path.join(out_dir, 'train.jsonl'), 'a') as train:
            with open(os.path.join(out_dir, 'test.jsonl'), 'a') as test:
                with open(os.path.join(out_dir, 'validate.jsonl'), 'a') as validate:
                    for example in self:
                        progress.update(1)
                        rand = random.random()
                        if rand < test_prop:
                            test.write(json.dumps(example) + '\n')
                        if rand < test_prop + val_prop:
                            validate.write(json.dumps(example) + '\n')
                        else:
                            train.write(json.dumps(example) + '\n')
