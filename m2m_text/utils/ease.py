"""
Created on 2021/04/05
@author Sangwoo Han
"""
import numpy as np

from ..datasets._base import Dataset


def get_ease_weight(dataset: Dataset, lamda: float):
    G = (dataset.y.T @ dataset.y).astype(np.float64)
    diag_indices = np.diag_indices(G.shape[0])
    G[diag_indices] += lamda
    P = np.linalg.inv(G.toarray())
    B = P / (-np.diag(P))
    B[diag_indices] = 0

    return B
