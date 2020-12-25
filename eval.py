"""Load data from jsonlines files into an ElasticSearch index."""

import sys
import re
import json
import time
from itertools import islice, chain
from pathlib import Path

from typing import Text, Dict, Any, Optional, List, Iterable

import tqdm
import numpy as np
from sklearn import metrics
from elasticsearch import Elasticsearch
from elasticsearch import AsyncElasticsearch

ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'pubmed_articles'


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


def msearch(client: Elasticsearch,
            queries: List[Text],
            index: Text,
            size: int = 50,
            fields: List[Text] = ['abstract']):
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


@timeit
def run_eval(client, k, page_size, fields: List[Text]):
    ex_batch_iter = edge_iter(client, limit=1000, page_size=page_size)
    y_true = []
    y_pred = []
    total = 0
    progress = tqdm.tqdm(unit="example")
    for ex_batch in ex_batch_iter:
        responses = msearch(client,
                    [ex['context'] for ex in ex_batch],
                    'pubmed_articles',
                    size=k,
                    fields=fields)['responses']
        progress.update(page_size)
        for i, response in enumerate(responses):
            results = response['hits']['hits']
            if not results:
                continue
            cited_pmid = ex_batch[i]['cited_pmid']
            y_t = [int(result["_source"].get('pmid') == cited_pmid) for result in results]
            y_t += [0] * (k - len(y_t))
            if any(y_t):
                total += 1
            y_true.append(y_t)
            y_p = [result["_score"] for result in results]
            y_p += [0] * (k - len(y_p))
            y_pred.append(y_p)
            if i % (page_size * 5) == 0:
                progress.set_postfix_str(f"nDCG: {metrics.ndcg_score(np.asarray(y_true), np.asarray(y_pred)):.3f}")
#                print(f"Query: {ex_batch[i]['context']}")
#                print(f"1: Abstract: {results[0]['_source']['abstract']}")
#                print(f"   Context: {results[0]['_source']['context']}")
#                print(f"2: Abstract: {results[1]['_source']['abstract']}")
#                print(f"   Context: {results[1]['_source']['context']}")
#                print(f"3: Abstract: {results[2]['_source']['abstract']}")
#                print(f"   Context: {results[2]['_source']['context']}")
    print("\n\n")
    print(f"nDCG: {metrics.ndcg_score(np.asarray(y_true), np.asarray(y_pred)):.3f}")
    print(f"A total of {total} correct results retrieved in top {k}.")


if __name__ == "__main__":

    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)

    run_eval(client, k=200, page_size=50, fields=['abstract^2', 'context^4', 'text'])
