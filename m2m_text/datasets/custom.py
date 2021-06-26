"""
Created on 2021/06/24
@author Sangwoo Han

Custom Dataset
"""
from typing import Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class CDataset(Dataset):
    def __init__(
        self,
        texts: np.ndarray,
        labels: Optional[csr_matrix] = None,
        is_train: bool = True,
    ):
        self.texts = texts
        self.labels = labels
        self.is_train = is_train

        if is_train and labels is None:
            raise ValueError("labels should be set when is_train is true")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.is_train:
            return (
                idx,
                self.texts[idx],
                torch.from_numpy(self.labels[idx].toarray().squeeze()),
            )

        return idx, self.texts[idx]


def collate_fn(batch):
    return (
        torch.LongTensor([b[0] for b in batch]),
        np.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
    )
