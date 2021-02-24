"""
Created on 2021/01/08
@author Sangwoo Han

Dataset for text format
"""
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..preprocessing import truncate_text
from ..utils import download_from_url, extract_archive
from ..utils.data import (
    get_emb_init,
    get_le,
    get_mlb,
    get_sparse_features,
    get_tokenized_texts,
    get_vocab,
)
from ._base import Dataset, TDataX, TDataXTensor, TDataY, TDataYTensor


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
        emb_init_filename (str): emb init filename
        maxlen (int): Max length of texts.
        pad (str): Pad token.
        unknwon (str): Unknwon token.
        tokenizer_model_name (str, optional): bert tokenizer model name.
        multi_label (bool): Set true if y is multi label.
    """

    def __init__(
        self,
        vocab_filename: str = "vocab.npy",
        tokenized_filename: str = "tokenized_texts.pkl",
        label_encoder_filename: str = "label_encoder",
        emb_init_filename: str = "emb_init.npy",
        maxlen: int = 500,
        pad: str = "<PAD>",
        unknown: str = "<UNK>",
        tokenizer_model_name: Optional[str] = None,
        multi_label: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_path = os.path.join(self.data_dir, vocab_filename)
        self.tokenized_path = os.path.join(
            self.data_dir, ("train_" if self.train else "test_") + tokenized_filename
        )
        self.sparse_path = os.path.join(
            self.data_dir, ("train_" if self.train else "test_") + "sparse.npz"
        )
        self.le_path = os.path.join(self.data_dir, label_encoder_filename)
        self.w2v_model_path = os.path.join(self.root, self.w2v_model)
        self.emb_init_path = os.path.join(self.data_dir, emb_init_filename)
        self.pad = pad
        self.unknown = unknown
        self.maxlen = maxlen
        self.tokenizer_model_name = tokenizer_model_name
        self.get_le = get_mlb if multi_label else get_le

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
        train_npz_path = os.path.join(self.data_dir, self.train_npz)
        npz_path = os.path.join(
            self.data_dir, self.train_npz if self.train else self.test_npz
        )

        train_npz_path, ext = os.path.splitext(train_npz_path)
        npz_path, _ = os.path.splitext(npz_path)

        if self.tokenizer_model_name:
            train_npz_path += "_" + self.tokenizer_model_name.replace("-", "_")
            npz_path += "_" + self.tokenizer_model_name.replace("-", "_")

        train_npz_path += f"_{self.maxlen}L" + ext
        npz_path += f"_{self.maxlen}L" + ext

        if not os.path.isfile(train_npz_path):
            if self.tokenizer_model_name:
                self._load_data_bert(train_npz_path)
            else:
                self._load_data(train_npz_path)

        return (
            self._load_data_bert(npz_path)
            if self.tokenizer_model_name
            else self._load_data(npz_path)
        )

    def _load_data_bert(self, npz_path: str) -> Tuple[TDataX, TDataY]:
        if os.path.isfile(npz_path):
            with np.load(npz_path, allow_pickle=True) as npz:
                input_ids, attention_mask, labels = (
                    npz["input_ids"],
                    npz["attention_mask"],
                    npz["labels"],
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_name)
            texts, labels = self.raw_data()

            input_ids = []
            attention_mask = []
            for text in tqdm(texts, desc="Tokenizing"):
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    return_tensors="np",
                    max_length=self.maxlen,
                )
                input_ids.append(inputs["input_ids"])
                attention_mask.append(inputs["attention_mask"])

            input_ids = np.concatenate(input_ids)
            attention_mask = np.concatenate(attention_mask)

            np.savez(
                npz_path,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        self._labels = labels

        le = self.get_le(self.le_path, labels)
        labels = le.transform(labels)

        return (input_ids, attention_mask), labels

    def _load_data(self, npz_path: str) -> Tuple[TDataX, TDataY]:
        """Load data, preprocessing if needed."""
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

        le = self.get_le(self.le_path, labels)
        labels = le.transform(labels)

        self.emb_init = get_emb_init(
            self.emb_init_path, self.vocab_path, self.w2v_model_path
        )

        return texts, labels

    def get_sparse_features(
        self, max_features: int = 100_000, force: bool = False
    ) -> csr_matrix:
        """Return sparse (tf-idf) features."""
        if not os.path.isfile(self.sparse_path):
            texts = None if os.path.isfile(self.tokenized_path) else self.raw_data()[0]
            tokenized_texts = get_tokenized_texts(self.tokenized_path, texts)
        else:
            tokenized_texts = None

        sparse_x = get_sparse_features(
            self.sparse_path, tokenized_texts, max_features=max_features, force=force
        )

        if self.split_indices is not None:
            sparse_x = sparse_x[self.split_indices]

        return sparse_x

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

        if not self.tokenizer_model_name:
            self.download_w2v_model()

        self.texts, self.labels = self.load_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[TDataXTensor, TDataYTensor]:
        if type(self.texts) == tuple:
            texts = tuple(torch.from_numpy(text[index]) for text in self.texts)
        else:
            texts = torch.from_numpy(self.texts[index])

        return texts, torch.from_numpy(self.labels[index, None].squeeze())

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


class DrugReviewSmall(DrugReview):
    base_folder = "drugReviewSmall"

    url = "https://drive.google.com/uc?id=1jc29iVmdyEDMRxngMoUTQ_PED_ySxeSz"

    filename = "drugReviewSmall.tar.gz"
    tgz_md5 = "0b74728596c617ea8993ceda71eaf6d0"

    file_list = [
        ("train.csv", "52ac786fd1bdc1891f61439aa6596260"),
        ("test.csv", "1bd45029abb30e7de9f60bbd0a25d529"),
    ]


class DrugReviewSmallv2(DrugReview):
    base_folder = "drugReviewSmallv2"

    url = "https://drive.google.com/uc?id=1NfjI481TaxHtcByijM5jAEXDK4arAZoc"

    filename = "drugReviewSmallv2.tar.gz"
    tgz_md5 = "f444c73417b55302896dab676e088419"

    file_list = [
        ("train.csv", "c44683ecbf08c41e3d6b14a76a79131e"),
        ("test.csv", "7ef03c50d824acf0385dac4ac4849ee2"),
    ]


class EURLex(TextDataset):
    """`EURLex Dataset.`

    Args:
        root (string, optional): Root directory of dataset. default: ./data
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set. default: True
        maxlen (int, optional): Maximum length of input text. default: 500
    """

    base_folder = "EURLex"

    url = "https://drive.google.com/uc?id=10A3_QWetbknuOpv4hK-F03T4hQGaufxO"

    filename = "EURLex.tar.gz"
    tgz_md5 = "d055ef38681cd6b78844e335c3bab1e7"

    file_list = [
        ("train_raw.npz", "62e50968bc5c469b6ac0270e27cc891d"),
        ("test_raw.npz", "87fef94edf237e2071fd1827618824c6"),
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
        super().__init__(root=root, train=train, multi_label=True, *args, **kwargs)
        self.download()

        if not self.tokenizer_model_name:
            self.download_w2v_model()

        self.texts, self.labels = self.load_data()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> Tuple[TDataXTensor, TDataYTensor]:
        if type(self.texts) == tuple:
            texts = tuple(torch.from_numpy(text[index]) for text in self.texts)
        else:
            texts = torch.from_numpy(self.texts[index])

        return texts, torch.from_numpy(self.labels[index].toarray().squeeze()).float()

    def raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        npz_path = os.path.join(
            self.data_dir, "train_raw.npz" if self.train else "test_raw.npz"
        )

        with np.load(npz_path, allow_pickle=True) as npz:
            return npz["texts"], npz["labels"]

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


class AmazonCat(TextDataset):
    """`AmazonCat Dataset.`

    Args:
        root (string, optional): Root directory of dataset. default: ./data
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set. default: True
        maxlen (int, optional): Maximum length of input text. default: 500
    """

    base_folder = "AmazonCat"

    url = "https://drive.google.com/uc?id=1HiWzrk1d-OX4pvjVdABX7Sf0ehKauEjB"

    filename = "AmazonCat.tar.gz"
    tgz_md5 = "4298b78dfeeabdc7f0ae694e1810b715"

    file_list = [
        ("train_raw.npz", "e0d159b51257b0f3b269711600cda8f4"),
        ("test_raw.npz", "3da6b6f59e4c56f6e8cc767b3ab50482"),
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
        super().__init__(root=root, train=train, multi_label=True, *args, **kwargs)
        self.download()

        if not self.tokenizer_model_name:
            self.download_w2v_model()

        self.texts, self.labels = self.load_data()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> Tuple[TDataXTensor, TDataYTensor]:
        if type(self.texts) == tuple:
            texts = tuple(torch.from_numpy(text[index]) for text in self.texts)
        else:
            texts = torch.from_numpy(self.texts[index])

        return texts, torch.from_numpy(self.labels[index].toarray().squeeze()).float()

    def raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        npz_path = os.path.join(
            self.data_dir, "train_raw.npz" if self.train else "test_raw.npz"
        )

        with np.load(npz_path, allow_pickle=True) as npz:
            return npz["texts"], npz["labels"]

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


class Wiki10(TextDataset):
    """`Wiki10 Dataset.`

    Args:
        root (string, optional): Root directory of dataset. default: ./data
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set. default: True
        maxlen (int, optional): Maximum length of input text. default: 500
    """

    base_folder = "Wiki10"

    url = "https://drive.google.com/uc?id=12DJFss2AxFoy6cF14UbdnrksDZarl8Ay"

    filename = "Wiki10.tar.gz"
    tgz_md5 = "9736461bce798bacf215179a1381717a"

    file_list = [
        ("train_raw.npz", "93ecf58d863c2fa9ff7da1519fa199cf"),
        ("test_raw.npz", "860fcd37c4d062747f512948cab75686"),
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
        super().__init__(root=root, train=train, multi_label=True, *args, **kwargs)
        self.download()

        if not self.tokenizer_model_name:
            self.download_w2v_model()

        self.texts, self.labels = self.load_data()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> Tuple[TDataXTensor, TDataYTensor]:
        if type(self.texts) == tuple:
            texts = tuple(torch.from_numpy(text[index]) for text in self.texts)
        else:
            texts = torch.from_numpy(self.texts[index])

        return texts, torch.from_numpy(self.labels[index].toarray().squeeze()).float()

    def raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        npz_path = os.path.join(
            self.data_dir, "train_raw.npz" if self.train else "test_raw.npz"
        )

        with np.load(npz_path, allow_pickle=True) as npz:
            return npz["texts"], npz["labels"]

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
