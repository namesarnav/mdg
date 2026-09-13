from typing import List, Optional
import random
from mdg.datamodels import TimeDocument, SpatialDocument


class DatasetLoader:
    def __init__(self, **kwargs):
        self.config = kwargs

    def first_n(self, n: int, split: Optional[str] = None, **kwargs) -> dict[str, Optional[List]]:
        """
        Load the first n examples from the dataset split (train/dev/test)
        :param n: Number of examples to load
        :param split: Dataset split to load
        :return: a dict with keys 'train', 'dev', 'test' and values as lists of documents or None if not available
        """
        self.train, self.dev, self.test = self.run(split, **kwargs)
        d = {}
        if n is not None and n > 0:
            if self.train is not None and split == "train":
                d["train"] = self.train[:n]
            if self.dev is not None and split == "dev":
                d["dev"] = self.dev[:n]
            if self.test is not None and split == "test":
                d["test"] = self.test[:n]
        if split is not None:
            return {split: d.get(split, None)}
        return d

    def sample(self, n: int, split: Optional[str] = None, **kwargs) -> dict[str, Optional[List]]:
        """
        Sample n examples from the dataset split (train/dev/test)
        :param n: Number of examples to sample
        :param split: Dataset split to sample from
        :return: a dict with keys 'train', 'dev', 'test' and values as lists of documents or None if not available
        """
        self.train, self.dev, self.test = self.run(split, **kwargs)
        d = {}
        if n is not None and n > 0:
            if self.train is not None and split == "train":
                d["train"] = random.sample(self.train, min(n, len(self.train)))
            if self.dev is not None and split == "dev":
                d["dev"] = random.sample(self.dev, min(n, len(self.dev)))
            if self.test is not None and split == "test":
                d["test"] = random.sample(self.test, min(n, len(self.test)))
        if split is not None:
            return {split: d.get(split, None)}
        return d

    def run(self, split: Optional[str] = None, **kwargs) -> dict[str, Optional[List]]:
        """
        Load the dataset split (train/dev/test)
        :param split: Dataset split to load
        :return: a dict with keys 'train', 'dev', 'test' and values as lists of documents or None if not available
        """
        pass


from mdg.dataset_loaders.temporal import *
from mdg.dataset_loaders.spatial import *
from mdg.dataset_loaders.causal import *
