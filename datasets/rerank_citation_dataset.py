from torch.utils.data import Dataset
import jsonlines
from typing import Text


class RerankCitationDataset(Dataset):

    def __init__(self,
                 fname: Text,
                 preprocess=None):
        super().__init__()

        self.preprocess = preprocess
        with jsonlines.open(fname) as reader:
            self.data = list(reader)
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
