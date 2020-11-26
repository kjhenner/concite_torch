"""Load data from jsonlines files into an ElasticSearch index."""

import sys
import re
from pathlib import Path

from typing import Text, Dict, Any, Optional

import jsonlines
import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'pubmed_articles'


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


def create_es_index(client: Elasticsearch,
                    index_name: Text,
                    b: float = 0.75,
                    k1: float = 1.2) -> None:
    """Create an ElasticSearch index for storing articles."""
    client.indices.delete(index=index_name, ignore=[400, 404])
    body = {"settings": {"index": {"similarity": {"default": {"type": "BM25",
                                                             "b": 0.75,
                                                             "k1": 1.2}}}},
            "mappings": {"properties": {"abstract": {"type": "text"},
                                        "year": {"type": "integer"},
                                        "id": {"type": "keyword"},
                                        "aggregate_contexts": {"type": "text"}}}}
    return client.indices.create(index=index_name, body=body, ignore=400)


def to_es_entry(article_datum: Dict[Text, Any], _id: str):
    """Convert an article dict to the format expected for ES."""
    return {"abstract": article_datum.get("abstract"),
            "year": extract_year(article_datum.get("year")),
            "title": article_datum.get("title"),
            "journal-id": article_datum.get("title"),
            "id": _id,
            "pmid": article_datum.get("pmid")}


def generate_actions(data_dir: Text,
                     index_name: Text,
                     pattern: Text = r'.*_articles.jsonl',
                     limit: Optional[int] = None):
    """Given a data directory, yield articles and index entries."""
    files = [f_name for f_name in Path(data_dir).iterdir() if re.match(pattern, str(f_name))]
    idx = 0
    for fname in files:
        with jsonlines.open(fname) as reader:
            for article_datum in reader:
                if limit and idx == limit:
                    break
                yield {"index": {"_index": index_name, "_id": str(idx)}}
                yield to_es_entry(article_datum, str(idx))
                idx += 1


def count_articles(data_dir: Text,
                   pattern: Text = r'.*_articles.jsonl'):
    """Given a data directory, count the number of article entries."""
    files = [f_name for f_name in Path(data_dir).iterdir() if re.match(pattern, str(f_name))]
    total = 0
    for fname in files:
        with open(fname) as f:
            total += sum(1 for _ in f)
    return total


def search(client: Elasticsearch, query: Text, index: Text):
    body = {
        "query": {
            "match": {
                "abstract": query
            }
        }
    }
    client.search(body=body, index=index)


if __name__ == "__main__":

    data_dir = sys.argv[1]

    limit = None

    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)

    print("Creating ES index...")
    create_es_index(client, 'pubmed_articles')

    print("Counting article total...")
    total = count_articles(data_dir)
    if limit:
        total = min(total, limit)

    print("Indexing articles entries...")
    progress = tqdm.tqdm(unit="articles", total=total)
    successes = 0
    action_generator = generate_actions(data_dir, "pubmed_articles", limit=limit)
    update_alternate = 0
    for ok, action in streaming_bulk(client=client,
                                     index="pubmed_articles",
                                     actions=action_generator):
        # We insert an index and article for each article,
        # so only update tqdm every other time.
        if update_alternate:
            update_alternate = 0
            progress.update(1)
            successes += ok
        else:
            update_alternate = 1


    print(f"Indexed {successes/total} articles")
