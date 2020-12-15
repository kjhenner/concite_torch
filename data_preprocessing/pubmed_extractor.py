import sys
import os
import pprint
import json
import itertools
import csv
import jsonlines
import tarfile
import re
import multiprocessing as mp
import io
import tqdm

from collections import defaultdict
import xml.etree.cElementTree as etree

from nltk.tokenize import sent_tokenize

from typing import Text, List, Tuple, Dict, Any, Optional


def tar_iter(path: Text) -> Text:
    """Iterate through the contents of files in a tar.gz archive."""
    tar = tarfile.open(path, 'r:gz')
    for member in tar:
        f = tar.extractfile(member)
        if f:
            yield f.read().decode('utf-8')


def tar_count(path: Text, limit: Optional[int] = None) -> Text:
    """Count the files in a tar.gz archive."""
    total = 0
    for _ in tarfile.open(path, 'r:gz'):
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


def parse_batch(batch: List[Text]) -> List[Tuple[Dict[Text, Any]]]:
    """Parse a batch of document strings."""
    results = []
    for item in batch:
        try:
            results.append(parse_cit_contexts(item))
        except Exception as e:
            print(e)
    return results


def parse_cit_contexts(string: Text) -> Tuple[Dict[Text, Any]]:
    """Extract document and citation data from an XML document string."""
    tree = etree.iterparse(io.StringIO(initial_value=string), events=('start', 'end'))
    document = {}
    in_body = False
    following_xref = False
    prev_xref_rid = None
    offset = 0
    text = ''
    section = ''
    rid = ''
    ref_dict = defaultdict(dict)
    offset_dict = defaultdict(list)
    authors = []
    document['keywords'] = ''
    for event, elem in tree:
        if event == 'start':
            if elem.tag == 'body':
                in_body = True
            if elem.text is not None and elem.tag !='xref' and in_body:
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
                document['year'] = elem.text
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
                    ref_dict[rid]['year'] = elem.text
                if elem.tag == 'pub-id' and elem.get('pub-id-type') == 'pmid':
                    ref_dict[rid]['pmid'] = elem.text
                if elem.tag == 'name':
                    ref_dict[rid]['authors'] = ref_dict[rid].get('authors', [])
                    ref_dict[rid]['authors'].append({child.tag: child.text for child in elem})
                    authors.append({child.tag: child.text for child in elem})
        if event == 'end' and in_body == True:
            if elem.tag == 'body':
                in_body = False
            if elem.tag == 'title':
                title = elem.itertext()
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


def clean_anchor_remnants(text: Text):
    return re.sub(r'\s*[\[\(][\s\-,–;(and)]*[\]\)]', '', text)


def clean_authors(text: Text, authors: List[Dict[Text, Text]]):
    text = re.sub(r'et al\.', ' ', text)
    for author in authors:
        if author.get('surname'):
            text = re.sub(re.escape(author['surname']), '', text)
    return text


def get_edges(offset_dict, ref_dict, document, text, char_window=600):
    edges = defaultdict(dict)
    for rid, offsets in offset_dict.items():
        # Some papers are simply missing bibliography ref entries for some
        # xrefs
        if ref_dict.get(rid):
            for offset in offsets:
                authors = ref_dict[rid].get('authors', [])
                window = get_context_window(text, offset, char_window)
                window = clean_anchor_remnants(clean_authors(window, authors))
                context = mid_sentence(window)
                edges[context]['citing_paper_id'] = document.get('pmid')
                edges[context]['context'] = context
                if edges[context].get('cited_papers'):
                    edges[context]['cited_papers'].append(ref_dict[rid])
                else:
                    edges[context]['cited_papers'] = [ref_dict[rid]]
    return list(edges.values())


def write_edge_data(edge_data, out_path):
    with jsonlines.open(out_path, 'w') as writer:
        writer.write_all(edge_data)


def write_document_data(document_data, out_path):
    with jsonlines.open(out_path, 'w') as writer:
        writer.write_all(document_data)


def mid_sentence(string):
    mid = len(string)/2
    pos = 0
    for sent in sent_tokenize(re.sub(r'\[\.', ']. ', string)):
        if pos + len(str(sent)) > mid:
            return str(sent)
        else:
            pos += len(str(sent))


def batch(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


def parse_archive(archive_path: Text,
                  out_dir: Text,
                  batch_size: int = 18,
                  limit: Optional[int] = None,
                  pool_size: int = 12):
    pool = mp.Pool(pool_size)
    name = '.'.join(os.path.split(data_path)[-1].split('.')[:2])

    results = []

    print("counting files")
    total = tar_count(data_path, limit)

    progress = tqdm.tqdm(unit="documents", total=total)
    for i, res in enumerate(pool.imap_unordered(parse_batch,
                            batch(itertools.islice(tar_iter(data_path), total), batch_size))):
        results += res
        progress.update(batch_size)
    pool.close()

    articles = [x[0] for x in results if x[0].get('pmid')]
    edges = [edge for edge_list in results for edge in edge_list[1]]

    write_edge_data(edges, os.path.join(out_dir, f"{name}_edges.jsonl"))
    write_document_data(articles, os.path.join(out_dir, f"{name}_articles.jsonl"))


if __name__ == "__main__":
    data_path = sys.argv[1]
    out_dir = sys.argv[2]
    batch_size = int(sys.argv[3])
    parse_archive(data_path, out_dir, batch_size, limit=None)
