from pathlib import Path
from typing import Optional
import re

from torchtext.data import TabularDataset, Field, BucketIterator

import numpy as np

class PubMedCitronDataset(TabularDataset):

    name = 'pubmedcitron'

    def __init__(self,
            path: Text,
            path_filter: Text,
            text_field: Field,
            label_field: Field,
            **kwargs) -> None:
        """Create a PubMed Citron dataset instance given a path and fields.

        Arguments:
            path (Text): Path to the dataset's highest level directory.
            path_filter (Text): Pattern to filter paths in the directory.
            text_field (Text): The field that will be used for citation context text.
            label_field (Text): The target citation.
            Remaining keywords arguments are passed to the constructor of data.TabularDataset
        """

        fields = [('text', text_field), ('label', label_field)]
        examples = []

        super(PubMedCitronDataset, self).__init__(examples, fields, **kwargs)

    @classmethods
    def splits(cls,
            text_field: Field,
            label_field: Field,
            root: Text,
            train: Text='train',
            test: Text='test',
            validation: Text='validation',
            **kwargs) -> PubmedCitronDataset:
        """Create dataset object for splits of the IMDB dataset.
        path (Text): Path to the dataset's highest level directory.
        path_filter (Text): Pattern to filter paths in the directory.

        Arguments:
            text_field (Text): The field that will be used for citation context text.
            label_field (Text): The target citation.
            root (Text): Root dataset storage directory.
            train (Test): The directory that contains training examples.
            test (Text): The directory that contains test examples.
            validation (Text): The directory that contains validation examples.
            Remaining keywords arguments are passed to the constructor of data.TabularDataset
        """

        return super(PubMedCitronDataset, cls).splits(
                root=root, text_field=text_field, label_field=label_field,
                train=train, test=test, validation=validation, **kwargs)

    @classmethod
    def iters(cls,
        batch_size: int=32,
        device: Optional[int]=0,
        **kwargs) -> PubMedCitronDataset:
        """Create iterator objects for splits of the Pubmed Citron dataset.

        Arguments:
            batch_size (int): Batch size for the iterator
            device (int): Cuda device ID to create batches on. -1 to use CPU,
                to use the currently active GPU device.
            Remaining keyword arguments are passed to the splits class method.
        """

citation_text = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=False)
citation_target = Field(sequential=False, use_vocab=True)

train_data, test_data = CitationRetrievalDataset.splits(
        path='mydata',
        train='train.json',
        test='test.json',
        validation='validation.json',
        fields=fields
