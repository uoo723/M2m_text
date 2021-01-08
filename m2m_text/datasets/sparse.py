"""
Created on 2021/01/08
@author Sangwoo Han

Dataset for sparse feature (e.g. tf-idf)
"""

"""
Created on 2020/12/31
@author Sangwoo Han
"""

import os
from typing import Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix

from ..utils.data import get_le
from ._base import Dataset, TDataX, TDataY


class SparseDataset(Dataset):
    # Subclass must define these class members

    train_npz = None  # train npz filename
    test_npz = None  # test npz filename

    """Base class for sparse feature datasets

    Args:
        label_encoder_filename (str): Label encoder filename
    """

    def __init__(self, label_encoder_filename: str = "label_encoder", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.le_path = os.path.join(self.data_dir, label_encoder_filename)

    def load_data(self) -> Tuple[TDataX, TDataY]:
        npz_path = os.path.join(
            self.data_dir, self.train_npz if self.train else self.test_npz
        )

        with np.load(npz_path, allow_pickle=True) as npz:
            x = csr_matrix((npz["data"], npz["indices"], npz["indptr"]))
            labels = npz["y"]

        le = get_le(self.le_path, labels)
        self._labels = labels

        return x, le.transform(labels)

    @property
    def raw_y(self) -> TDataY:
        return self._labels

    @raw_y.setter
    def raw_y(self, labels):
        self._labels = labels


class RCV1(SparseDataset):
    """`RCV1 <https://https://archive.ics.uci.edu/ml/datasets/Reuters+RCV1+RCV2+Multilingual,+Multiview+Text+Categorization+Test+collection>`_ Dataset.

    Args:
        root (string, optional): Root directory of dataset. default: ./data
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set. default: True
    """

    base_folder = "rcv1"

    url = "https://drive.google.com/uc?id=1oLaGER_HEDIwAg89S92_WLbAMCRIinJY"

    filename = "rcv1.tar.gz"
    tgz_md5 = "5a122af1c6331f2276b0f20c59334a3f"

    file_list = [
        ("train.npz", "de6b46472356af41fa389b041c0dfb80"),
        ("test.npz", "6b35bc2922c4b5b73182e069d80fc3a3"),
    ]

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

        self._x, self._y = self.load_data()

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self._x[index].toarray()), torch.from_numpy(
            self._y[index, None].squeeze()
        )

    @classmethod
    def splits(cls, *args, **kwargs):
        return super().splits(*args, **kwargs)

    @property
    def x(self) -> TDataX:
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self) -> TDataY:
        return self._y

    @y.setter
    def y(self, y):
        self._y = y
