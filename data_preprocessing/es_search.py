"""Load data from jsonlines files into an ElasticSearch index."""

import sys
import re
import json
from pathlib import Path

from typing import Text, Dict, Any, Optional

import jsonlines
import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'pubmed_articles'


def search(client: Elasticsearch, query: Text, index: Text):
    body = {
        "query": {
            "match": {
                "context": query
            }
        }
    }
    return client.search(body=body, index=index)


if __name__ == "__main__":

    query = sys.argv[1]

    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)

    print(json.dumps(search(client, query, 'pubmed_articles'), indent=2))
