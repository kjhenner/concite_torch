from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import re

class ImageDataset(Dataset):

    def __init__(self, image_dir, match_pattern, limit=None):
        super().__init__()

        self.files = [f_name for f_name in Path(image_dir).iterdir() if re.match(match_pattern, str(f_name))]
        if limit:
            self.files = self.files[:limit]

    def __len__(self):
        return len(self.files)

    def open_as_array(self, idx):
        raw_img = np.array(Image.open(self.files[idx]))

        return (raw_img / np.iinfo(raw_img.dtype).max)

    def __getitem__(self, idx):
        return torch.tensor(self.open_as_array(idx), dtype=torch.float32)

    def __repr__(self):
        return f'Dataset class with {self.__len__()} files'
