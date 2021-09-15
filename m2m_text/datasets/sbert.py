"""
Created on 2021/07/12
@author Sangwoo Han
"""
import os
import pickle
import time
from datetime import timedelta
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from logzero import logger
from m2m_text.datasets.text import TextDataset
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BertDataset(Dataset):
    """Warpping Dataset class for BERT"""

    def __init__(
        self,
        dataset: TextDataset,
        model_name: str,
        verbose: bool = True,
        **tokenizer_kwargs,
    ) -> None:
        self.dataset = dataset
        self.verbose = verbose

        self.train = self.dataset.train

        self.npz_path = "train_" if self.train else "test_"
        self.npz_path += model_name.replace("-", "_")
        self.npz_path += f"_{tokenizer_kwargs['max_length']}L.npz"
        self.npz_path = os.path.join(self.dataset.data_dir, self.npz_path)

        self.input_ids, self.attention_mask = self._load_data(
            model_name, **tokenizer_kwargs
        )

    def _load_data(
        self, model_name: str, **tokenizer_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if os.path.isfile(self.npz_path):
            with np.load(self.npz_path) as npz:
                return torch.from_numpy(npz["input_ids"]), torch.from_numpy(
                    npz["attention_mask"]
                )

        with open(self.dataset.tokenized_path, "rb") as f:
            texts = pickle.load(f)

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.verbose:
            logger.info("Tokenize...")
            start = time.time()

        inputs = tokenizer([" ".join(s) for s in texts], **tokenizer_kwargs)

        if self.verbose:
            elapsed = time.time() - start
            logger.info(
                f"Finish Tokenization. {elapsed:.2f}s {timedelta(seconds=elapsed)}"
            )

        np.savez(
            self.npz_path,
            input_ids=inputs["input_ids"].numpy(),
            attention_mask=inputs["attention_mask"].numpy(),
        )

        return inputs["input_ids"], inputs["attention_mask"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            torch.from_numpy(self.y[idx].toarray().squeeze()).float(),
        )

    @property
    def raw_y(self) -> np.ndarray:
        return self.dataset.raw_y

    @property
    def y(self) -> csr_matrix:
        return self.dataset.y


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


def collate_fn2(batch):
    return (
        {
            "input_ids": torch.stack([b[0] for b in batch]),
            "attention_mask": torch.stack([b[1] for b in batch]),
        },
        torch.stack([b[2] for b in batch]),
    )
