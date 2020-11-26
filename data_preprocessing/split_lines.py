import sys
import random
import os
import jsonlines
import json
import re
import gzip

from collections import defaultdict
from itertools import chain
from typing import Text, Dict, Any, List

def iter_lines(fpaths: List[Text]) -> Dict[Text, Any]:
    for path in fpaths:
        print(path)
        with jsonlines.open(path) as reader:
            for obj in reader:
                yield obj

def clean_text(text: Text):
    return re.sub(r'\[?\s*?\d+\s*\-?\s*\d*\s*\]?', '', text)

def train_dev_test_split(input_dir: Text,
                         pattern: Text,
                         train_prop: float,
                         test_prop: float,
                         validation_prop: float,
                         output_dir: Text,
                         buffer_size: int=100000) -> None:

    fpaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if re.match(pattern, f)]
    total = sum([os.stat(fpath).st_size for fpath in fpaths])
    i = 0
    train_buff = ''
    test_buff = ''
    validation_buff = ''
    with open(os.path.join(output_dir, 'train.tsv'), 'w') as train,\
         open(os.path.join(output_dir, 'test.tsv'), 'w') as test,\
         open(os.path.join(output_dir, 'validation.tsv'), 'w') as validation:
        for line in iter_lines(fpaths):
            if not line.get('cited_paper', {}).get('pmid'):
                continue
            if not line.get('citing_paper_id'):
                continue
            if not line.get('cite_sentence'):
                continue
            i += 1
            line = '\t'.join([line['citing_paper_id'],
                              line['cited_paper']['pmid'],
                              clean_text(line['cite_sentence'])]) + '\n'
            rn = random.uniform(0,1)
            if rn < train_prop:
                train_buff += line
            elif rn < train_prop + test_prop:
                test_buff += line
            else:
                validation_buff += line
            if i % buffer_size == 0:
                train.write(train_buff)
                train_buff = ''
                test.write(test_buff)
                test_buff = ''
                validation.write(validation_buff)
                validation_buff = ''
        train.write(train_buff)
        train_buff = ''
        test.write(test_buff)
        test_buff = ''
        validation.write(validation_buff)
        validation_buff = ''

if __name__ == "__main__":
    input_dir = sys.argv[1]
    train_prop = float(sys.argv[2])
    test_prop = float(sys.argv[3])
    validation_prop = float(sys.argv[4])
    output_dir = sys.argv[5]
    if train_prop + test_prop + validation_prop != 1:
        raise Exception("train_prop, test_prop, and validation_prop must sum to 1")
    pattern = r'.*edges\.jsonl$'

    train_dev_test_split(input_dir, pattern, train_prop, test_prop, validation_prop, output_dir)
