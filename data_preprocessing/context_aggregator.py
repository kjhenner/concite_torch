"""Load data from jsonlines files into an ElasticSearch index."""

import sys
import os
import re
from pathlib import Path
from collections import defaultdict

from typing import Text, Dict, Any, Optional

import jsonlines
import tqdm


def extract_year(string: Text) -> Text:
    """Extract the 4 digit year from a year string.

    Note that the string extracted from the XML is very inconsistent.
    We have to expect that for many examples, the accurate year simply
    isn't reliably recoverable. However, these errors represent a relatively
    small proportion of the overall corpus.
    """
    if not string:
        return 0
    m = re.match(r'\d{4}', string)
    if m:
        return m[0]


def aggregate_contexts(data_dir: Text,
                       edge_count: int,
                       pattern: Text = r'.*_edges.jsonl'):
    """Given a data yield full contexts and cutoff contexts."""
    aggregates = defaultdict(list)
    files = [f_name for f_name in Path(data_dir).iterdir() if re.match(pattern, str(f_name))]
    progress = tqdm.tqdm(unit="edge", total=edge_count)
    for fname in files:
        with jsonlines.open(fname) as reader:
            for edge in reader:
                pmid = edge.get("citing_paper_id")
                if pmid:
                    for cited_paper in edge.get('cited_papers'):
                        if cited_paper.get('pmid'):
                            data = {"citing_paper_id": pmid,
                                    "context": edge["context"]}
                            aggregates[cited_paper['pmid']].append(data)
                            progress.update(1)
    return [{"pmid": k, "contexts": v} for k, v in aggregates.items()]


def count_edges(data_dir: Text,
                pattern: Text = r'.*_edges.jsonl'):
    """Given a data directory, count the number of edge entries."""
    files = [f_name for f_name in Path(data_dir).iterdir() if re.match(pattern, str(f_name))]
    total = 0
    for fname in files:
        with open(fname) as f:
            total += sum(1 for _ in f)
    return total


if __name__ == "__main__":

    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    cutoff_date = 2017

    print("Counting edges")
    edge_count = count_edges(data_dir)
    print(f"Counted {edge_count} edges")

    aggregates = aggregate_contexts(data_dir, edge_count)
 
    out_path = os.path.join(out_dir, "contexts.jsonl")
    with jsonlines.open(out_path, 'w') as writer:
            writer.write_all(aggregates)
