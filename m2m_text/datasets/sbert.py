"""
Created on 2021/07/12
@author Sangwoo Han
"""
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class SBertDataset(Dataset):
    """Warpping Dataset class for senteice BERT"""

    def __init__(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: Optional[csr_matrix],
        train: bool = True,
    ) -> None:
        self.inputs = inputs
        self.labels = labels
        self.is_train = train

        if train and labels is None:
            raise ValueError("labels should be set when is_train is true")

    def __len__(self) -> int:
        return self.inputs["input_ids"].shape[0]

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if self.is_train:
            return (
                idx,
                self.inputs["input_ids"][idx],
                self.inputs["attention_mask"][idx],
                torch.from_numpy(self.labels[idx].toarray().squeeze()),
            )
        else:
            return (
                idx,
                self.inputs["input_ids"][idx],
                self.inputs["attention_mask"][idx],
            )


def collate_fn(batch):
    if len(batch[0]) == 4:
        return (
            torch.LongTensor([b[0] for b in batch]),
            {
                "input_ids": torch.stack([b[1] for b in batch]),
                "attention_mask": torch.stack([b[2] for b in batch]),
            },
            torch.stack([b[3] for b in batch]),
        )
    else:
        return (
            torch.LongTensor([b[0] for b in batch]),
            {
                "input_ids": torch.stack([b[1] for b in batch]),
                "attention_mask": torch.stack([b[2] for b in batch]),
            },
        )
