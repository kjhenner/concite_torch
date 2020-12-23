"""Load data from jsonlines files into an ElasticSearch index."""

import sys
import re
import json
import time
from pathlib import Path

from typing import Text, Dict, Any, Optional, List

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


def paged_msearch(client: Elasticsearch,
                  queries: List[Text],
                  index: Text,
                  page_size: int = 20,
                  size: int = 50,
                  field: Text = 'abstract'):
    while queries:
        page_queries = queries[:page_size]
        queries = queries[page_size:]
        for response in msearch(client, page_queries, index, size, field)['responses']:
            yield response


def msearch(client: Elasticsearch,
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


def paged_edge_iter(client: Elasticsearch,
                    size: int = 500):
    data = client.search(index='pubmed_edges',
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


@timeit
def run_eval(client, examples, page_size, field: Text='abstract'):
    y_true = []
    y_pred = []
    total = 0
    k = 150
    progress = tqdm.tqdm(unit="example", total=len(examples))
    search = paged_msearch(client,
                           [ex[0] for ex in examples],
                           'pubmed_articles',
                           size=k,
                           page_size=page_size,
                           field=field)
    for i, response in enumerate(search):
        cited_pmid = examples[i][1]
        results = response['hits']['hits']
        if not results:
            continue
        y_t = [int(result["_source"].get('pmid') == cited_pmid) for result in results]
        y_t += [0] * (k - len(y_t))
        if any(y_t):
            total += 1
        y_true.append(y_t)
        y_p = [result["_score"] for result in results]
        y_p += [0] * (k - len(y_p))
        y_pred.append(y_p)
        progress.update(1)
        if i % page_size == 0:
            try:
                progress.set_postfix_str(f"nDCG: {metrics.ndcg_score(np.asarray(y_true), np.asarray(y_pred)):.3f}")
            except:
                import pdb; pdb.set_trace()
    print("\n\n")
    print(f"nDCG: {metrics.ndcg_score(np.asarray(y_true), np.asarray(y_pred)):.3f}")
    print(f"A total of {total} correct results retrieved in top {k}.")


if __name__ == "__main__":

    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)

    pmid_set = get_internal_pmid_set(data_dir)

    limit = 500

    examples = list(generate_examples(data_dir,
                                      limit=limit,
                                      pmid_set=pmid_set))

    run_eval(client, examples, 100, 'abstract')
    run_eval(client, examples, 100, 'context')
