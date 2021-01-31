from torch.utils.data import Dataset
from typing import List


class DictDataset(Dataset):
    """Simple dataset to represent data already loaded as a list of dicts."""

    def __init__(self,
                 data: List,
                 preprocess):
        super().__init__()

        self.preprocess = preprocess
        self.data = data

    def __getitem__(self, idx: int):
        if self.preprocess:
            return self.preprocess(self.data[idx])
        else:
            return self.data[idx]

    def __repr__(self):
        return 'Dataset class for data already loaded into memory as a dict.'

    def __len__(self):
        return len(self.data)
