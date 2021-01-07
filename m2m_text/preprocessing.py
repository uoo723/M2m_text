"""
Created on 2020/12/31
@author Sangwoo Han
"""
from collections import Counter
from typing import Callable, Iterable, Optional, Union

import numpy as np
from gensim.models import KeyedVectors
from logzero import logger
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from sklearn.utils import column_or_1d
from sklearn.utils._encode import _unique
from sklearn.utils.validation import _num_samples, check_is_fitted
from tqdm.auto import tqdm

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


def build_vocab(
    texts: Iterable[Iterable[str]],
    w2v_model: Optional[Union[KeyedVectors, str]] = None,
    pad: str = "<PAD>",
    unknown: str = "<UNK>",
    vocab_size: int = 500000,
    freq_times: int = 1,
    max_times: int = 1,
) -> np.ndarray:
    """Build vocab

    Args:
        texts (iter[iter[str]]): List of tokenized texts.
        w2v_model (KeyedVectors or str, optional): Pretrained w2v model
        pad (str): Pad token
        unknwon (str): Unknwon token
        vocab_size (int): Maximum number of vocab. default: 500,000
        tokenizer (func): Tokenizer. default nltk.tokenize.word_tokenize
        lower (bool): Lower text. default: True
        freq_times (int): Minimum number of tokens to be added to vocab. default: 1
        max_times (int): Maximum number of tokens to be added to vocab. default: 1

    Returns:
        vocab (np.npdarray): Numpy array of vocab.
            e.g. vocab[0] = 'token1'
    """
    if w2v_model is not None and isinstance(w2v_model, str):
        w2v_model = KeyedVectors.load(w2v_model)

    vocab = [pad, unknown]
    counter = Counter(token for text in texts for token in text)

    for word, freq in sorted(
        counter.items(),
        key=lambda x: (x[1], x[0] in w2v_model if w2v_model else True),
        reverse=True,
    ):
        if (w2v_model is not None and word in w2v_model) or freq >= freq_times:
            vocab.append(word)

        if freq < max_times or vocab_size == len(vocab):
            break

    return np.array(vocab)


def truncate_text(
    token_ids: Iterable[int],
    maxlen: int = 500,
    padding_idx: int = 0,
    unknown_idx: int = 1,
) -> np.ndarray:
    """Truncate text

    Args:
        token_ids (iter[int]): List of token ids.
        maxlen (int): Maximum length of tokens. default: 500
        padding_idx (int): padding index.
        unknown_idx (int): unknown index.

    Returns:
        padded_token_ids (np.ndarray): Padded token id 1-D numpy array.
    """
    token_ids = np.array(
        [
            list(x[:maxlen]) + [padding_idx] * (maxlen - len(x))
            for x in tqdm(token_ids, desc="Truncating text.")
        ]
    )
    token_ids[(token_ids == padding_idx).all(axis=1), 0] = unknown_idx
    return token_ids


def tokenize(
    texts: Iterable[str],
    tokenizer: Callable = word_tokenize,
    lower: bool = True,
) -> Iterable[Iterable[str]]:
    """Tokenize texts.

    Args:
        texts: Iterable[str],
        tokenizer (func): Tokenizer. default nltk.tokenize.word_tokenize
        lower (bool): Lower text. default: True

    Returns:
        tokenized (iter[iter[str]]): List of tokenized texts.
    """
    import nltk

    nltk.download("punkt", quiet=True)

    logger.info("Tokenizing texts.")
    tokenized_texts = []
    for text in tqdm(texts, desc="Tokenizing"):
        text = text.lower() if lower else text
        tokenized_texts.append(tokenizer(text))

    return tokenized_texts
