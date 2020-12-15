"""Load data from jsonlines files into an ElasticSearch index."""

import sys
import re
from pathlib import Path

from typing import Text, Dict, Any, Optional, List

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
    #client.indices.delete(index=index_name, ignore=[400, 404])
    client.indices.delete(index=index_name)
    body = {
             "settings": {
               "index": {
                 "similarity": {
                   "default": {
                     "type": "BM25",
                             "b": 0.75,
                             "k1": 1.2
                   }
                 }
               },
               "analysis": {
                 "analyzer": {
                   "default": {
                     "type": "standard",
                     "stopwords": "_english_"
                   }
                 }
               }
             },
             "mappings": {
               "properties": {
                 "abstract": {"type": "text"},
                 "year": {"type": "integer"},
                 "title": {"type": "text"},
                 "journal_id": {"type": "text"},
                 "text": {"type": "text"},
                 "pmid": {"type": "keyword"},
                 "citation_contexts": {"type": "text"}
               }
             }
           }
    return client.indices.create(index=index_name, body=body, ignore=400)


def to_es_entry(article_datum: Dict[Text, Any], context: List[Text]):
    """Convert an article dict to the format expected for ES."""
    return {"_id": article_datum.get("pmid"),
            "abstract": article_datum.get("abstract"),
            "year": extract_year(article_datum.get("year")),
            "title": article_datum.get("title"),
            "text": article_datum.get("text"),
            "journal-id": article_datum.get("journal-id"),
            "context": context,
            "pmid": article_datum.get("pmid")}


def generate_actions(data_dir: Text,
                     index_name: Text,
                     context_lookup: Dict[Text, Any],
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
                pmid = article_datum.get("pmid")
                if pmid:
                    yield to_es_entry(article_datum, context_lookup.get(pmid, []))
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


def bulk_index_articles(client: Elasticsearch,
                        context_lookup: Dict[Text, Any],
                        limit: Optional[int] = None):

    print("Creating ES index...")
    create_es_index(client, 'pubmed_articles')

    print("Counting article total...")
    total = count_articles(data_dir)
    if limit:
        total = min(total, limit)

    print("Indexing articles entries...")
    progress = tqdm.tqdm(unit="article", total=total)
    successes = 0
    action_generator = generate_actions(data_dir,
                                        "pubmed_articles",
                                        context_lookup,
                                        limit=limit)
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


def load_context_lookup(path: Text):
    context_lookup = {}
    print(f"Loading context lookup")
    with jsonlines.open(path) as reader:
        for datum in reader:
            context = [context['context'] for context in datum["contexts"]]
            context_lookup[datum["pmid"]] = context
    return context_lookup


if __name__ == "__main__":

    data_dir = sys.argv[1]
    context_path = sys.argv[2]

    limit = None

    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)

    context_lookup = load_context_lookup(context_path)
    bulk_index_articles(client, context_lookup, limit)
