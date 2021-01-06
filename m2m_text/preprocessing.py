"""
Created on 2020/12/31
@author Sangwoo Han
"""
import numpy as np
from sklearn import preprocessing
from sklearn.utils import column_or_1d
from sklearn.utils._encode import _unique
from sklearn.utils.validation import _num_samples, check_is_fitted


UNKNOWN = "<unknown>"


class _unknowndict(dict):
    """Dictionary with support for missing."""

    def __missing__(self, key):
        return len(self) - 1


class LabelEncoder(preprocessing.LabelEncoder):
    """Extend sklearn.preprocessing.LabelEncoder to handle unknown class"""

    def fit(self, y):
        """Fit label encoder.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = np.concatenate([_unique(y), np.array([UNKNOWN])])
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        return self.fit(y).transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        table = _unknowndict({val: i for i, val in enumerate(self.classes_)})

        return np.array([table[v] for v in y])
