"""
Created on 2021/01/08
@author Sangwoo Han
"""
import os
from typing import Union

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from ..utils import check_integrity, download_from_url, extract_archive

TDataX = Union[np.ndarray, csr_matrix]
TDataY = Union[np.ndarray, csr_matrix]


class Dataset(torch.utils.data.Dataset):
    # Subclass must define these class members

    base_folder = None  # Base folder name for this dataset
    url = None  # Download url
    filename = None  # Downloaded archive filename
    tgz_md5 = None  # md5 for archive file

    file_list = None  # File list and md5 pairs in archive.
    # e.g. [('train.csv', '9b820c0b4a1aae30ff80a1538a4b3f0d')]

    """Base class for Dataset

    Args:
        root (str): Root directory for dataset
        train (bool): If true, train dataset will be loaded,
            otherwise test dataset will be loadded.
    """

    def __init__(self, root: str = "./data", train: bool = True) -> None:
        self.root = os.path.expanduser(root)
        self.data_dir = os.path.join(self.root, self.base_folder)
        self.train = train

    def download(self, quiet: bool = False) -> None:
        """Download archive from url

        Args:
            quiet (bool): If true, quiet downloading
        """
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
            train_dataset (Dataset): Splitted train dataset.
            valid_dataset (Dataset): Splitted test datasets.
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
    def x(self) -> TDataX:
        """Input"""
        raise NotImplemented

    @property
    def y(self) -> TDataY:
        """Encoded output"""
        raise NotImplemented

    @property
    def raw_y(self) -> TDataY:
        """Raw output (not encoded)"""
        raise NotImplemented
