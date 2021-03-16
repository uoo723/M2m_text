"""
Created on 2021/01/08
@author Sangwoo Han
"""
from .sparse import RCV1
from .text import (
    AmazonCat,
    DrugReview,
    DrugReviewSmall,
    DrugReviewSmallv2,
    EURLex,
    EURLex4K,
    Wiki10,
)

__all__ = [
    "DrugReview",
    "RCV1",
    "DrugReviewSmall",
    "DrugReviewSmallv2",
    "EURLex",
    "AmazonCat",
    "Wiki10",
]
