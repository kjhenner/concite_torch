"""Load data from jsonlines files into an ElasticSearch index."""

import sys
import re
import json
from pathlib import Path
from itertools import islice, chain

from typing import Text, Dict, Any, Optional, List

import tqdm
import multiprocessing as mp
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'pubmed_articles'


def msearch_pmids(client: Elasticsearch,
                  pmids: List[Text],
                  year_cutoff: int = None):
    search_arr = []
    for pmid in pmids:
        header = {'index': 'pubmed_edges'}
        search_arr.append(header)
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"cited_pmids": pmid}}
                    ]
                }
            },
            "_source": ["context"]
        }
        if year_cutoff:
            body["query"]["bool"]["filter"] = ({
                "range": {"year": {"lte": year_cutoff}}
            })
        search_arr.append(body)
    request = ''
    for item in search_arr:
        request += f"{json.dumps(item)} \n"
    return zip(pmids, client.msearch(request, request_timeout=30)["responses"])


def paged_article_iter(client: Elasticsearch,
                       size: int = 500):
    data = client.search(index=ES_INDEX_NAME,
                         scroll='10m',
                         size=size,
                         body={})
    sid = data['_scroll_id']
    scroll_size = len(data['hits']['hits'])
    while scroll_size:
        yield [hit['_source']['pmid'] for hit in data['hits']['hits']]
        data = client.scroll(scroll_id=sid, scroll='10m')
        sid = data['_scroll_id']
        scroll_size = len(data['hits']['hits'])


def to_article_update(pmid: Text, contexts: Text):
    return {
        "_op_type": "update",
        "_index": "pubmed_articles",
        "_id": pmid,
        "doc": {
            "context": contexts
        }
    }


def update_action_iter(client: Elasticsearch):
    for pmids in paged_article_iter(client):
        results = list(msearch_pmids(client, pmids, year_cutoff=2017))
        for pmid, hits in results:
            contexts = [hit["_source"]["context"] for hit in hits["hits"]["hits"]]
            yield to_article_update(pmid, contexts)


def batch_iterable(iterable, n):
    """Batch an iterable into batches of size n."""
    it = iter(iterable)
    for first in it:
        yield list(chain([first], islice(it, n-1)))


def bulk_update(action_batch):
    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)
    bulk(client=client,
         index='pubmed_articles',
         actions=action_batch)


if __name__ == "__main__":

    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)

    batch_size = 2000
    pool_size = 15
    pool = mp.Pool(pool_size)

    total = client.count(index="pubmed_articles")['count']
    progress = tqdm.tqdm(unit="document", total=total)
    for _ in pool.imap_unordered(bulk_update, batch_iterable(update_action_iter(client), batch_size)):
        progress.update(batch_size)
    pool.close()
