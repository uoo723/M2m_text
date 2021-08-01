"""
Created on 2021/06/24
@author Sangwoo Han

Custom Dataset
"""
from typing import Tuple, Union

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class IDDataset(Dataset[T_co]):
    """Prepend idx"""

    dataset: Dataset[T_co]

    def __init__(
        self,
        dataset: Dataset[T_co],
    ) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, idx: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        idx = torch.tensor(idx) if type(idx) == int else idx
        return (idx, *self.dataset[idx])
