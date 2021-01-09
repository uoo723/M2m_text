"""
Created on 2021/01/08
@author Sangwoo Han

Dataset for text format
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..preprocessing import truncate_text
from ..utils import download_from_url, extract_archive
from ..utils.data import get_le, get_tokenized_texts, get_vocab
from ._base import Dataset, TDataX, TDataY


class TextDataset(Dataset):
    # Subclass must define these class members

    w2v_model = None  # gensim w2v model name"glove.840B.300d.gensim"
    w2v_model_url = None  # gensim w2v model download url
    w2v_model_tgz_md5 = None  # gensim w2v model archive md5

    train_npz = None  # train npz filename
    test_npz = None  # test npz filename

    """Base class for text datasets

    Args:
        vocab_filename (str): Vocab filename.
        tokenized_filename (str): Tokenized text filename.
        label_encoder_filename (str): Label encoder filename
        maxlen (int): Max length of texts.
        pad (str): Pad token.
        unknwon (str): Unknwon token.
    """

    def __init__(
        self,
        vocab_filename: str = "vocab.npy",
        tokenized_filename: str = "tokenized_texts.pkl",
        label_encoder_filename: str = "label_encoder",
        maxlen: int = 500,
        pad: str = "<PAD>",
        unknown: str = "<UNK>",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_path = os.path.join(self.data_dir, vocab_filename)
        self.tokenized_path = os.path.join(
            self.data_dir, ("train_" if self.train else "test_") + tokenized_filename
        )
        self.le_path = os.path.join(self.data_dir, label_encoder_filename)
        self.w2v_model_path = os.path.join(self.root, self.w2v_model)
        self.pad = pad
        self.unknown = unknown
        self.maxlen = maxlen

    def download_w2v_model(self, quiet: bool = False) -> None:
        """Download w2v model

        Args:
            quiet (bool): If true, quiet downloading
        """
        fpath = os.path.join(self.root, self.w2v_model + ".tar.gz")
        os.makedirs(self.root, exist_ok=True)

        file_list = [self.w2v_model, self.w2v_model + ".vectors.npy"]

        if all(map(lambda x: os.path.isfile(os.path.join(self.root, x)), file_list)):
            return

        fpath = download_from_url(
            self.w2v_model_url, fpath, self.w2v_model_tgz_md5, quiet
        )

        if fpath:
            extract_archive(fpath)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data, preprocessing if needed."""
        npz_path = os.path.join(
            self.data_dir, self.train_npz if self.train else self.test_npz
        )
        npz_path = os.path.splitext(npz_path)
        npz_path = npz_path[0] + f"_{self.maxlen}L" + npz_path[1]  # e.g. train_500L.npz

        if os.path.isfile(npz_path):
            with np.load(npz_path, allow_pickle=True) as npz:
                texts, labels = npz["texts"], npz["labels"]
        else:
            texts, labels = self.raw_data()
            tokenized_texts = get_tokenized_texts(self.tokenized_path, texts)
            vocab = get_vocab(
                self.vocab_path,
                tokenized_texts,
                self.w2v_model_path,
                pad=self.pad,
                unknown=self.unknown,
            )
            stoi = {word: i for i, word in enumerate(vocab)}
            texts = np.array(
                [
                    [stoi.get(token, stoi[self.unknown]) for token in text]
                    for text in tqdm(tokenized_texts, desc="Converting token to id")
                ],
                dtype=np.object,
            )

            texts = truncate_text(
                texts, self.maxlen, stoi[self.pad], stoi[self.unknown]
            )

            np.savez(npz_path, texts=texts, labels=labels)

        self._labels = labels

        le = get_le(self.le_path, labels)
        labels = le.transform(labels)
        return texts, labels

    def raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrun raw data for preprocessing"""
        raise NotImplemented

    @property
    def raw_y(self) -> TDataY:
        return self._labels

    @raw_y.setter
    def raw_y(self, labels):
        self._labels = labels


class DrugReview(TextDataset):
    """`DrugReview <https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018>`_ Dataset.

    Args:
        root (string, optional): Root directory of dataset. default: ./data
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set. default: True
        maxlen (int, optional): Maximum length of input text. default: 500
    """

    base_folder = "drugReview"

    url = "https://drive.google.com/uc?id=1_UMpS9X76KYuCeJQY6HVKm2pPi-wxofy"

    filename = "drugReview.tar.gz"
    tgz_md5 = "b12cd9fe6ad124887b7d759e7a0bc7ae"

    file_list = [
        ("train.csv", "9b820c0b4a1aae30ff80a1538a4b3f0d"),
        ("test.csv", "c356cba00149b0e8ada3e413a91af483"),
    ]

    w2v_model = "glove.840B.300d.gensim"
    w2v_model_url = "https://drive.google.com/uc?id=1q9n32NeCVuCJpZK-aoH48-iExW2Hf9nS"
    w2v_model_tgz_md5 = "baa5434ab9d5833b56805dc12f0094a0"

    train_npz = "train.npz"
    test_npz = "test.npz"

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(root=root, train=train, *args, **kwargs)
        self.download()
        self.download_w2v_model()

        self.texts, self.labels = self.load_data()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.texts[index]), torch.from_numpy(
            self.labels[index, None].squeeze()
        )

    def raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        csv_path = os.path.join(
            self.data_dir, "train.csv" if self.train else "test.csv"
        )

        df = pd.read_csv(csv_path)
        return df["review"].to_numpy(), df["condition"].to_numpy()

    @classmethod
    def splits(cls, *args, **kwargs):
        return super().splits(*args, **kwargs)

    @property
    def x(self) -> TDataX:
        return self.texts

    @x.setter
    def x(self, x):
        self.texts = x

    @property
    def y(self) -> TDataY:
        return self.labels

    @y.setter
    def y(self, y):
        self.labels = y
