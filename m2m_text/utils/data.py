"""
Created on 2021/01/07
@author Sangwoo Han
"""
import os
import pickle
from collections import Counter
from typing import Iterable, Optional, Union

import joblib
import numpy as np
from gensim.models import KeyedVectors
from torch.utils.data import Dataset

from ..preprocessing import LabelEncoder, build_vocab, tokenize


def get_word_emb(
    w2v_model: Union[KeyedVectors, str],
    vocab: np.ndarray,
    pad: str = "<PAD>",
    unknown: str = "<UNK>",
) -> np.ndarray:
    """Get pretrained word embedding

    Args:
        w2v_model (KeyedVectors or str): Pretrained w2v model
        vocab (np.ndarray): 1-D vocab array
        pad (str): Pad token
        unknwon (str): Unknwon token

    Returns:
        embedding (np.ndarray) -> Emebedding 2-D numpy array.
    """
    if w2v_model is not None and isinstance(w2v_model, str):
        w2v_model = KeyedVectors.load(w2v_model)

    emb_init = []
    emb_size = w2v_model.vector_size

    for token in enumerate(vocab):
        if token in [pad, unknown] or token not in w2v_model:
            emb_init.append(np.random.uniform(-1.0, 1.0, emb_size))
        else:
            emb_init.append(w2v_model[token])

    return np.array(emb_init)


def get_le(
    le_path: str, labels: Optional[Iterable] = None, force: bool = False
) -> LabelEncoder:
    if os.path.isfile(le_path) and not force:
        return joblib.load(le_path)
    le = LabelEncoder()
    le.fit(labels)
    joblib.dump(le, le_path)
    return le


def get_tokenized_texts(
    tokenized_path: str, texts: Optional[Iterable[str]] = None, force: bool = False
) -> Iterable[Iterable[str]]:
    if os.path.isfile(tokenized_path) and not force:
        with open(tokenized_path, "rb") as f:
            return pickle.load(f)

    tokenized_texts = tokenize(texts)
    with open(tokenized_path, "wb") as f:
        pickle.dump(tokenized_texts, f)

    return tokenized_texts


def get_emb_init(
    emb_init_path: str,
    vocab_path: Optional[str] = None,
    w2v_model_path: Optional[str] = None,
    force: bool = False,
) -> np.ndarray:
    if os.path.isfile(emb_init_path) and not force:
        return np.load(emb_init_path)

    vocab = get_vocab(vocab_path)
    emb_init = get_word_emb(w2v_model_path, vocab)
    np.save(emb_init_path, emb_init)

    return emb_init


def get_vocab(
    vocab_path: str,
    tokenized_texts: Optional[Iterable[Iterable[str]]] = None,
    w2v_model_path: Optional[str] = None,
    pad: str = "<PAD>",
    unknown: str = "<UNK>",
    force: bool = False,
) -> np.ndarray:
    if os.path.isfile(vocab_path) and not force:
        return np.load(vocab_path, allow_pickle=True)

    vocab = build_vocab(tokenized_texts, w2v_model_path, pad=pad, unknown=unknown)
    np.save(vocab_path, vocab)
    return vocab


def get_oversampled_data(
    dataset: Dataset, num_sample_per_class: Iterable[int]
) -> Iterable[int]:
    """Return a list of imbalanced indices from a dataset.

    # Reference: https://github.com/alinlab/M2m/blob/master/data_loader.py

    Args:
        dataset (Dataset): Dataset object.
        num_samples_per_class (iter[int]): number of samples per class.

    Returns:
        oversampled_list (iter[int]): Oversampled weight list with shape (n_samples,)
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = list(num_sample_per_class)

    selected_list = []
    indices = list(range(0, length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_samples[label])
            num_sample_per_class[label] -= 1

    return selected_list


def get_n_samples_per_class(y: np.ndarray):
    """Returns num of samples of class

    Args:
        y (np.ndarray): 1-D numpy array of class.

    Returns:
        n_samples_per_class (np.ndarray): Number of samples per class 1-D array.
            number samples for class i is n_samples_per_calss[i]
    """
    cnt = Counter(y)
    n_samples_per_class = np.zeros(len(cnt))

    for i, count in cnt.items():
        n_samples_per_class[i] = count

    return n_samples_per_class
