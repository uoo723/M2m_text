"""
Created on 2021/01/05
@author Sangwoo Han
"""
import hashlib
import os
from collections import Counter
from typing import Any, Callable, Iterable, Optional, Union

import gdown
import numpy as np
from gensim.models import KeyedVectors
from logzero import logger
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    """Calculate file md5

    Reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py

    Args:
        fpath (str): File path
        chunk_size (int): Chunk size to be read

    Returns:
        md5 (str): Calculated md5
    """
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    """Check md5 between file and input md5

    Reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py

    Args:
        fpath (str): File path
        md5 (str): md5 string to compare
        kwargs: Argument for `calculate_md5` func

    Returns:
        check (bool): Returns True if matched, otherwise False.
    """
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    """Check file integrity

    Reference: https://github.com/pytorch/vision/blob/master/torchvision/datasets/utils.py

    Args:
        fpath (str): File path
        md5 (str, optional): md5 to compare

    Returns:
        check (str): Returns True if passed integrity, otherwise False.
    """
    if not os.path.isfile(fpath):
        return False

    if md5 is None:
        return True

    return check_md5(fpath, md5)


def download_from_url(
    url: str, fpath: str, md5: Optional[str] = None, quiet: bool = False
) -> str:
    """Download archive from url

    Args:
        url (str): URL for dowload
        fpath (str): output path
        md5 (str, optional): md5 to check integrity
        quiet (bool): quiet downloading

    Returns:
        fpath (str): output path
    """
    if os.path.exists(fpath):
        return fpath

    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    fpath = gdown.cached_download(
        url,
        fpath,
        md5=md5,
    )

    return fpath


def extract_archive(fpath: str) -> bool:
    """Extract archive

    Args:
        fpath (str): Archive path

    Returns:
        success (bool): Returns True if succeed, otherwise False
    """
    if not os.path.exists(fpath):
        return False

    gdown.extractall(fpath)

    return True


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
