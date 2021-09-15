"""
Created on 2021/01/08
@author Sangwoo Han
"""
from .custom import IDDataset
from .sbert import BertDataset, SBertDataset
from .sparse import RCV1
from .text import (
    Amazon_670K,
    AmazonCat,
    AmazonCat13K,
    DrugReview,
    DrugReviewSmall,
    DrugReviewSmallv2,
    EURLex,
    EURLex4K,
    Wiki10,
    Wiki10_31K,
)

__all__ = [
    "DrugReview",
    "RCV1",
    "DrugReviewSmall",
    "DrugReviewSmallv2",
    "EURLex",
    "EURLex4K",
    "AmazonCat",
    "Wiki10",
    "AmazonCat13K",
    "Wiki10_31K",
    "SBertDataset",
    "IDDataset",
    "Amazon_670K",
]

__all__ += ["BertDataset", "SBertDataset"]
