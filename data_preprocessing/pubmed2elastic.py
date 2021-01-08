import os
import jsonlines
import tarfile
import re
import multiprocessing as mp
import io
import tqdm
import argparse
import logging
from pathlib import Path

from collections import defaultdict
from itertools import islice, chain

import xml.etree.cElementTree as etree

from nltk.tokenize import sent_tokenize

from typing import Text, List, Tuple, Dict, Any, Optional, Iterator

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

ES_HOST = 'localhost'
ES_PORT = 9200
ES_ARTICLE_INDEX_NAME = 'pubmed_articles'
ES_EDGE_INDEX_NAME = 'pubmed_edges'

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def batch_iterable(iterable, n):
    """Batch an iterable into batches of size n."""
    it = iter(iterable)
    for first in it:
        yield list(chain([first], islice(it, n-1)))


def tar_iter(path: Text) -> Text:
    """Iterate through the contents of files in a tar.gz archive."""
    tar = tarfile.open(path, 'r:gz')
    for member in tar:
        f = tar.extractfile(member)
        if f:
            yield f.read().decode('utf-8')


def tar_count(path: Text, limit: Optional[int] = None, progress=True) -> Text:
    """Count the files in a tar.gz archive."""
    logger.info(f"Counting files in {path}...")
    total = 0
    if progress:
        it = tqdm.tqdm(tarfile.open(path, 'r:gz'), unit='documents')
    else:
        it = tarfile.open(path, 'r:gz')
    for _ in it:
        total += 1
        if total == limit:
            return total
    return total


def get_context_window(text: Text, offset: int, window_size: int) -> Text:
    """Given a string, get a window around a specified offset point."""
    start = max([offset - window_size, 0])
    end = min([offset + window_size, len(text)])
    return text[start:end]


def rid2int(rid: Text) -> int:
    """Extract the integer value from a PubMed reference id (rid) tag."""
    m = re.match(r'(C|CIT|c|r|cr|CR|cit|ref|b|bibr|R|B)(\d+).*', rid)
    if m is not None and m.group(2) is not None:
        return int(m.group(2))
    else:
        m = re.match(r'.*(C|CIT|c|r|cr|CR|cit|ref|b|bibr|R|B)[\-_]?(\d+)$', rid)
        if m is not None and m.group(2) is not None:
            return int(m.group(2))


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
    else:
        return None


def clean_anchor_remnants(text: Text):
    return re.sub(r'\s*[\[\(][\s\-,–;(and)]*[\]\)]', '', text)


def clean_authors(text: Text, authors: List[Dict[Text, Text]]):
    text = re.sub(r'et al\.', ' ', text)
    for author in authors:
        if author.get('surname'):
            text = re.sub(re.escape(author['surname']), '', text)
    return text


def parse_cit_contexts(string: Text) -> Tuple[Dict[Text, Any]]:
    """Extract document and citation data from an XML document string."""
    tree = etree.iterparse(io.StringIO(initial_value=string), events=('start', 'end'))
    document = {}
    in_body = False
    prev_xref_rid = None
    offset = 0
    text = ''
    rid = ''
    ref_dict = defaultdict(dict)
    offset_dict = defaultdict(list)
    authors = []
    document['keywords'] = ''
    for event, elem in tree:
        if event == 'start':
            if elem.tag == 'body':
                in_body = True
            if elem.text is not None and elem.tag != 'xref' and in_body:
                offset += len(elem.text)
                text += elem.text
            if elem.tag == 'journal-id' and elem.get('journal-id-type') == 'nlm-ta':
                document['journal-id'] = elem.text
            if elem.tag == 'article-id' and elem.get('pub-id-type') == 'pmc':
                document['pmc'] = elem.text
            if elem.tag == 'article-id' and elem.get('pub-id-type') == 'pmid':
                document['pmid'] = elem.text
            if elem.tag == 'article-title':
                document['title'] = ' '.join(elem.itertext())
            if elem.tag == 'abstract':
                document['abstract'] = ' '.join(elem.itertext())
            if elem.tag == 'year':
                document['year'] = extract_year(elem.text)
            if elem.tag == 'kwd':
                document['keywords'] += ' ' + str(elem.text)
            if elem.tag == 'ref':
                rid = rid2int(elem.get('id'))
            if rid != '':
                if elem.tag == 'source':
                    ref_dict[rid]['source'] = ''.join(elem.itertext())
                if elem.tag == 'volume':
                    ref_dict[rid]['volume'] = ''.join(elem.itertext())
                if elem.tag == 'fpage':
                    ref_dict[rid]['fpage'] = ''.join(elem.itertext())
                if elem.tag == 'lpage':
                    ref_dict[rid]['lpage'] = ''.join(elem.itertext())
                if elem.tag == 'article-title':
                    ref_dict[rid]['title'] = ''.join(elem.itertext())
                if rid and elem.tag == 'year':
                    ref_dict[rid]['year'] = extract_year(elem.text)
                if elem.tag == 'pub-id' and elem.get('pub-id-type') == 'pmid':
                    ref_dict[rid]['pmid'] = elem.text
                if elem.tag == 'name':
                    ref_dict[rid]['authors'] = ref_dict[rid].get('authors', [])
                    ref_dict[rid]['authors'].append({child.tag: child.text for child in elem})
                    authors.append({child.tag: child.text for child in elem})
        if event == 'end' and in_body:
            if elem.tag == 'body':
                in_body = False
            if elem.tag == 'xref' and elem.get('ref-type') == 'bibr':
                for id_part in elem.get('rid').split(' '):
                    offset_dict[rid2int(id_part)].append(offset)
                    if prev_xref_rid:
                        for id in range(prev_xref_rid + 1, rid2int(id_part)):
                            offset_dict[id].append(offset)
            if elem.tail is not None:
                offset += len(elem.tail) + 1
                text += elem.tail + ' '
            if elem.tag == 'xref' and elem.get('ref-type') == 'bibr' and elem.tail == '-':
                prev_xref_rid = rid2int(elem.get('rid'))
            else:
                prev_xref_rid = None
    # Strip citation anchor remnants.
    document['text'] = clean_anchor_remnants(clean_authors(text, authors))
    return (document, get_edges(offset_dict, ref_dict, document, text))


def get_edges(offset_dict, ref_dict, document, text, char_window=600):
    edges = []
    for rid, offsets in offset_dict.items():
        # Some papers are simply missing bibliography ref entries for some
        # xrefs
        if ref_dict.get(rid) and ref_dict[rid].get('pmid'):
            for offset in offsets:
                authors = ref_dict[rid].get('authors', [])
                window = get_context_window(text, offset, char_window)
                window = clean_anchor_remnants(clean_authors(window, authors))
                context = mid_sentence(window)
                edges.append({
                    'position': offset / len(text),
                    'citing_pmid': document.get('pmid'),
                    'year': extract_year(document.get('year')),
                    'context': context,
                    'cited_pmid': ref_dict[rid]['pmid']
                })
    return edges


def mid_sentence(string):
    mid = len(string)/2
    pos = 0
    for sent in sent_tokenize(re.sub(r'\[\.', ']. ', string)):
        if pos + len(str(sent)) > mid:
            return str(sent)
        else:
            pos += len(str(sent))


def load_batch(text_batch: Iterator):
    """Load a batch of parsed document results into ES."""
    skipped = defaultdict(int)
    results = []
    for doc_text in text_batch:
        try:
            results.append(parse_cit_contexts(doc_text))
        except Exception as e:
            logger.exception('Error', exc_info=e)
            skipped['parse_error'] += 1
    doc_data = [to_es_article_entry(x[0]) for x in results if x[0].get('pmid')]
    for x in results:
        if x[0].get('pmid'):
            doc_data.append(to_es_article_entry(x[0]))
        else:
            skipped['no_pmid'] += 1
    edge_data = [to_es_edge_entry(edge) for edge_list in results for edge in edge_list[1]]
    client = Elasticsearch(
        hosts=[ES_HOST],
        scheme='http',
        verify_certs=False,
        port=ES_PORT)
    bulk(client=client,
         index='pubmed_articles',
         actions=doc_data)
    bulk(client=client,
         index='pubmed_edges',
         actions=edge_data)
    return skipped


def create_article_index(client: Elasticsearch,
                         index_name: Text,
                         b: float = 0.75,
                         k1: float = 1.2) -> None:
    """Create an ElasticSearch index for storing articles."""
    client.indices.delete(index=index_name, ignore=[400, 404])
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
                 "context": {"type": "text"}
               }
             }
           }
    return client.indices.create(index=index_name, body=body, ignore=400)


def create_edge_index(client: Elasticsearch,
                      index_name: Text,
                      b: float = 0.75,
                      k1: float = 1.2) -> None:
    """Create an ElasticSearch index for storing edges."""
    client.indices.delete(index=index_name, ignore=[400, 404])
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
                 "year": {"type": "integer"},
                 "context": {"type": "text"},
                 "predicted_intent": {"type": "keyword"},
                 "position": {"type": "float"},
                 "citing_pmid": {"type": "keyword"},
                 "cited_pmid": {"type": "keyword"},
                 "internal": {"type": "boolean"}
               }
             }
           }
    return client.indices.create(index=index_name, body=body, ignore=400)


def to_es_article_entry(article_datum: Dict[Text, Any]):
    """Convert an article dict to the format expected for ES."""
    return {"_id": article_datum.get("pmid"),
            "abstract": article_datum.get("abstract"),
            "year": extract_year(article_datum.get("year")),
            "title": article_datum.get("title"),
            "text": article_datum.get("text"),
            "journal-id": article_datum.get("journal-id"),
            "context": None,
            "pmid": article_datum.get("pmid")}


def to_es_edge_entry(edge_datum: Dict[Text, Any]):
    """Convert an edge dict to the format expected for ES."""
    return {"year": extract_year(edge_datum.get("year")),
            "context": edge_datum.get("context"),
            "position": edge_datum.get("position"),
            "citing_pmid": edge_datum.get("citing_pmid"),
            "cited_pmid": edge_datum.get("cited_pmid")}


def count_articles(data_dir: Text,
                   pattern: Text = r'.*_articles.jsonl'):
    """Given a data directory, count the number of article entries."""
    files = [f_name for f_name in Path(data_dir).iterdir() if re.match(pattern, str(f_name))]
    total = 0
    for fname in files:
        with open(fname) as f:
            total += sum(1 for _ in f)
    return total


def load_context_lookup(path: Text):
    context_lookup = {}
    print("Loading context lookup")
    with jsonlines.open(path) as reader:
        for datum in reader:
            context = [context['context'] for context in datum["contexts"]]
            context_lookup[datum["pmid"]] = context
    return context_lookup


def parse_archive(archive_path: Text,
                  batch_size: int = 50,
                  limit: Optional[int] = None,
                  pool: int = 15):

    total = tar_count(archive_path, limit)

    progress = tqdm.tqdm(unit="document", total=total)
    batch_it = batch_iterable(islice(tar_iter(archive_path), total), batch_size)
    skipped = defaultdict(int)
    if pool > 1:
        pool = mp.Pool(pool)
        for status in pool.imap_unordered(load_batch, batch_it):
            for k, v in status.items():
                skipped[k] += v
            progress.update(batch_size)
            progress.set_postfix(skipped)
        pool.close()
    else:
        for batch in batch_it:
            status = load_batch(batch_it)
            for k, v in status.items():
                skipped[k] += v
            progress.update(batch_size)
            progress.set_postfix(skipped)


def path_iter(path):
    for f_name in Path(path).iterdir():
        if re.match(r'^.*\.xml\.tar\.gz$', str(f_name)):
            yield f_name
        else:
            print(f"{path} does not match the expected .xml.tar.gz file extension "
                  "for OA archive files. Skipping processing for this file.")


def main():
    parser = argparse.ArgumentParser(description='Load data from PMC OA to Elasticsearch.')
    parser.add_argument(
        'path',
        metavar='PATH',
        help='path to a directory containing PMC OA bulk xml.tar.gz files OR a single xml.tar.gz file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='total number of documents to parse per archive file',
        default=None,
    )
    parser.add_argument(
        '--pool',
        type=int,
        help='cpu pool for multiprocessing (set to 1 to disable multiprocessing)',
        default=mp.cpu_count() - 1
    )
    parser.add_argument(
        '--es_host',
        help='Elasticsearch host',
        default='localhost'
    )
    parser.add_argument(
        '--es_port',
        help='Elasticsearch port',
        default='9200'
    )
    parser.add_argument(
        '--progress',
        help='show progress bar',
        type=bool,
        default=True
    )

    args = parser.parse_args()

    client = Elasticsearch(
        hosts=[args.es_host],
        scheme='http',
        verify_certs=False,
        port=args.es_port)

    create_article_index(client, 'pubmed_articles')
    create_edge_index(client, 'pubmed_edges')

    if os.path.isdir(args.path):
        paths = list(path_iter(args.path))
        for i, f_name in enumerate(path_iter(args.path)):
            logger.info(f"Processing documents from {f_name}... ({i+1} of {len(paths)} archives)")
            parse_archive(f_name, limit=args.limit)
    elif re.match(r'^.*\.xml\.tar\.gz$', args.path):
        logger.info(f"Processing documents from {args.path}...")
        parse_archive(args.path, limit=args.limit)
    else:
        logger.info(f"{args.path} does not match the expected .xml.tar.gz file extension "
                    "for OA archive files. Skipping processing for this file.")


if __name__ == "__main__":
    main()
