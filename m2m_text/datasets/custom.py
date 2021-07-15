"""
Created on 2021/06/24
@author Sangwoo Han

Custom Dataset
"""
from typing import Tuple

import torch
from torch.utils.data import Dataset


class IDDataset(Dataset):
    """Prepend idx"""

    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (torch.tensor(idx), *self.dataset[idx])
