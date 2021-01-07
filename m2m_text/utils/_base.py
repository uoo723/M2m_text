"""
Created on 2021/01/07
@author Sangwoo Han
"""
import hashlib
import os
from typing import Any, Optional

import gdown


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
