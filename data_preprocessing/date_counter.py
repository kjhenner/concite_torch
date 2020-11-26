import sys
import os
import re
import jsonlines
from collections import defaultdict
from typing import Text

def extract_year(string: Text) -> Text:
    if not string:
        return None
    m = re.match(r'\d{4}', string)
    if m:
        return m[0]

if __name__ == "__main__":
    data_dir = sys.argv[1]
    year_counts = defaultdict(int)

    for fpath in os.listdir(data_dir):
        if fpath.split('_')[-1] == "articles.jsonl":
            with jsonlines.open(os.path.join(data_dir, fpath)) as reader:
                for obj in reader:
                    extracted_year = extract_year(obj['year'])
                    if extracted_year:
                        year_counts[extracted_year] += 1

    year_counts = list(year_counts.items())
    total = sum([yc[1] for yc in year_counts])
    running_total = 0
    for yc in sorted(year_counts):
        print(yc)
        print(f"{float(running_total)/float(total)}")
        running_total += yc[1]
