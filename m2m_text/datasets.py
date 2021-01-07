"""
Created on 2020/12/31
@author Sangwoo Han
"""

import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .preprocessing import truncate_text
from .utils import check_integrity, download_from_url, extract_archive
from .utils.data import get_le, get_tokenized_texts, get_vocab


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str = "./data", train: bool = True) -> None:
        self.root = os.path.expanduser(root)
        self.data_dir = os.path.join(self.root, self.base_folder)
        self.train = train

    def download(self, quiet: bool = False) -> None:
        fpath = os.path.join(self.data_dir, self.filename)

        if all(
            map(
                lambda x: os.path.isfile(os.path.join(self.data_dir, x[0])),
                self.file_list,
            )
        ):
            return

        fpath = download_from_url(self.url, fpath, self.tgz_md5, quiet)

        if fpath:
            extract_archive(fpath)

        for filename, md5 in self.file_list:
            assert check_integrity(os.path.join(self.data_dir, filename), md5)

    @classmethod
    def splits(cls, test_size: Union[int, float], *args, **kwargs):
        """Splits dataset.

        Args:
            test_size (int | float): Test size or ratio.

        Returns:
            datasets (tuple[Dataset, Dataset]): Splitted datasets.
        """
        train_dataset = cls(*args, **kwargs)
        valid_dataset = cls(*args, **kwargs)
        train_x, valid_x, train_y, valid_y, train_raw_y, valid_raw_y = train_test_split(
            train_dataset.x, train_dataset.y, train_dataset.raw_y, test_size=test_size
        )

        train_dataset.x = train_x
        train_dataset.y = train_y
        valid_dataset.x = valid_x
        valid_dataset.y = valid_y
        train_dataset.raw_y = train_raw_y
        valid_dataset.raw_y = valid_raw_y

        return train_dataset, valid_dataset

    @property
    def x(self) -> np.ndarray:
        raise NotImplemented

    @property
    def y(self) -> np.ndarray:
        raise NotImplemented

    @property
    def raw_y(self) -> np.ndarray:
        raise NotImplemented


class TextDataset(Dataset):
    """Base class for text datasets

    Args:
        vocab_filename (str): Vocab filename.
        tokenized_filename (str): Tokenized text filename.
        label_encoder_filename (str): Label encoder filename
        pad (str): Pad token.
        unknwon (str): Unknwon token.
        maxlen (int): Max length of texts.
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

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return torch.from_numpy(texts), torch.from_numpy(labels)

    def raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplemented

    @property
    def raw_y(self):
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
        return self.texts[index], self.labels[index]

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
    def x(self) -> np.ndarray:
        return self.texts

    @x.setter
    def x(self, x):
        self.texts = x

    @property
    def y(self) -> np.ndarray:
        return self.labels

    @y.setter
    def y(self, y):
        self.labels = y
