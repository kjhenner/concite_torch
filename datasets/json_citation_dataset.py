from torch.utils.data import Dataset
import json
from typing import Dict


class JsonCitationDataset(Dataset):

    def __init__(self,
                 data: Dict,
                 preprocess=None):
        super().__init__()

        self.preprocess = preprocess
        self.data = data
        self.total = len(self.data)

    def __getitem__(self, idx: int):
        if self.preprocess:
            return self.preprocess(self.data[idx])
        else:
            return self.data[idx]

    def __len__(self):
        return self.total

    def __repr__(self):
        return 'Dataset class with for pubmed citation edges'
