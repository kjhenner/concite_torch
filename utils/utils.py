"""Utililty functions related to Elasticsearch."""

import json
import time
from itertools import islice, chain
from typing import Text, List

from torch.utils.data._utils.collate import default_collate

from elasticsearch import Elasticsearch


def collate_hits(tokenizer, query, hits, batch_size, max_length: int = 256):
    collated_hit_batches = []
    for batch in batch_iterable(hits, batch_size):
        tokenized_batch = []
        for hit in batch:
            tokenized = tokenizer(
                text=query,
                text_pair=hit['_source']['abstract'] or '',
                add_special_tokens=True,
                max_length=max_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True
            )
            tokenized['source'] = {
                'pmid': hit['_source']['pmid'],
                'abstract': hit['_source']['abstract'],
                'context': query
            }
            print("-"*20)
            for k, v in tokenized.items():
                print(f"{k}: {v}")
                if type(v) != dict:
                    print(v.shape)
            tokenized_batch.append(tokenized)
        print(len(tokenized_batch))
        collated_hit_batches.append(default_collate(tokenized_batch))
    return collated_hit_batches


def batch_iterable(iterable, n):
    """Batch an iterable into batches of size n."""
    it = iter(iterable)
    for first in it:
        yield list(chain([first], islice(it, n-1)))


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()

        print(f"{method.__name__} {(te - ts):.2f} s")
        return result
    return timed


def msearch_contexts(client: Elasticsearch,
                     queries: List[Text],
                     index: Text = 'pubmed_articles',
                     size: int = 50,
                     fields: List[Text] = ['abstract']):
    """Multi-search docs from a list of contexts."""
    search_arr = []
    for query in queries:
        header = {'index': index}
        search_arr.append(header)
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "most_fields"
                }
            },
            "size": size,
        }
        search_arr.append(body)
    request = ''
    for item in search_arr:
        request += f"{json.dumps(item)} \n"
    return client.msearch(request, request_timeout=30)


def edge_iter(client: Elasticsearch,
              page_size: int = 100,
              limit: int = None):
    total = 0
    body = {
        "query": {
            "bool": {
                "must": {"match": {"internal": True}},
                "filter": {"range": {"year": {"gt": 2017}}}
            }
        }
    }
    data = client.search(index='pubmed_edges',
                         scroll='10m',
                         size=page_size,
                         body=body)
    sid = data['_scroll_id']
    scroll_size = len(data['hits']['hits'])
    while scroll_size:
        yield [hit['_source'] for hit in data['hits']['hits']]
        total += page_size
        if limit and total >= limit:
            return
        data = client.scroll(scroll_id=sid, scroll='10m')
        sid = data['_scroll_id']
        scroll_size = len(data['hits']['hits'])
