"""Load data from jsonlines files into an ElasticSearch index."""

import sys
import re
import json
import asyncio
import time
from pathlib import Path

from typing import Text, Dict, Any, Optional, List

import jsonlines
import tqdm
import numpy as np
from sklearn import metrics
from elasticsearch import Elasticsearch
from elasticsearch import AsyncElasticsearch

ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'pubmed_articles'


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, *kwargs)
        te = time.time()

        print(f"{method.__name__} {(te - ts):.2f} s")
        return result
    return timed


def search(client: Elasticsearch,
           query: Text,
           index: Text,
           size: int = 50,
           field: Text = 'abstract'):
    body = {
        "query": {
            "match": {
                field: query
            }
        },
        "size": size,
    }
    return client.search(body=body, index=index)


async def paged_msearch(client: Elasticsearch,
                        queries: List[Text],
                        index: Text,
                        page_size: int = 50,
                        size: int = 50,
                        field: Text = 'abstract'):
    while queries:
        page_queries = queries[:page_size]
        queries = queries[page_size:]
        for response in msearch(client, page_queries, index, size, field)['responses']:
            yield response


async def msearch(client: Elasticsearch,
                  queries: List[Text],
                  index: Text,
                  size: int = 50,
                  field: Text = 'abstract'):
    search_arr = []
    for query in queries:
        header = {'index': index}
        search_arr.append(header)
        body = {
            "query": {
                "match": {
                    field: query
                }
            },
            "size": size,
        }
        search_arr.append(body)
    request = ''
    for item in search_arr:
        request += f"{json.dumps(item)} \n"
    return client.msearch(request, request_timeout=30)


def generate_examples(data_dir: Text,
                      pattern: Text = r'.*_edges.jsonl',
                      limit: Optional[int] = None,
                      pmid_set: Optional[set] = None):
    """Given a data directory, yield articles and index entries."""
    files = [f_name for f_name in Path(data_dir).iterdir() if re.match(pattern, str(f_name))]
    idx = 0
    for fname in files:
        with jsonlines.open(fname) as reader:
            for edge_datum in reader:
                if limit and idx == limit:
                    return
                if not pmid_set or edge_datum['cited_paper'].get('pmid') in pmid_set:
                    yield (edge_datum['cite_sentence'], edge_datum['cited_paper'].get('pmid'))
                    idx += 1


def get_internal_pmid_set(data_dir: Text,
                          pattern: Text = r'.*_articles.jsonl'):
    """Given a data directory, get the set of article PMIDs."""
    files = [f_name for f_name in Path(data_dir).iterdir() if re.match(pattern, str(f_name))]
    pmids = []
    progress = tqdm.tqdm(unit="files", total=len(files))
    for fname in files:
        with jsonlines.open(fname) as reader:
            for article_datum in reader:
                if article_datum['pmid']:
                    pmids.append(article_datum['pmid'])
            progress.update(1)
    return set(pmids)


def count_articles(data_dir: Text,
                   pattern: Text = r'.*_articles.jsonl'):
    """Given a data directory, count the number of article entries."""
    files = [f_name for f_name in Path(data_dir).iterdir() if re.match(pattern, str(f_name))]
    total = 0
    for fname in files:
        with open(fname) as f:
            total += sum(1 for _ in f)
    return total


def extract_year(string: Text) -> Text:
    """Extract the 4 digit year from a year string.

    Note that the string extracted from the XML is very inconsistent.
    We have to expect that for many examples, the accurate year simply
    isn't reliably recoverable. However, these errors represent a relatively
    small proportion of the overall corpus.
    """
    if not string:
        return None
    m = re.match(r'\d{4}', string)
    if m:
        return m[0]


@timeit
def run_eval(client, examples, page_size):
    y_true = []
    y_pred = []
    total = 0
    k = 50
    progress = tqdm.tqdm(unit="example", total=len(examples))
    search = paged_msearch(client,
                           [ex[0] for ex in examples],
                           'pubmed_articles',
                           size=k,
                           page_size=page_size)
    for i, response in enumerate(search):
        cited_pmid = examples[i][1]
        y_t = [int(result["_source"].get('pmid') == cited_pmid) for result in response['hits']['hits']]
        if any(y_t):
            total += 1
        y_true.append(y_t)
        y_pred.append([result["_score"] for result in response['hits']['hits']])
        progress.update(1)
        if i % page_size == 0:
            progress.set_postfix_str(f"nDCG: {metrics.ndcg_score(np.asarray(y_true), np.asarray(y_pred)):.3f}")
    print("\n\n")
    print(f"nDCG: {metrics.ndcg_score(np.asarray(y_true), np.asarray(y_pred)):.3f}")
    print(f"A total of {total} correct results retrieved in top {k}.")

if __name__ == "__main__":

    data_dir = sys.argv[1]

    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)


    client = AsyncElasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)

    #pmid_set = get_internal_pmid_set(data_dir)
    pmid_set = None

    limit = 1000

    examples = list(generate_examples(data_dir,
                                      limit=limit,
                                      pmid_set=pmid_set))

    run_eval(client, examples, 200)
