"""
Created on 2021/01/05
@author Sangwoo Han
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def get_accuracy(labels: np.ndarray, targets: np.ndarray):
    correct_mask = labels == targets
    correct_total = len(targets)
    correct = correct_mask.sum()

    c_matrix = confusion_matrix(labels, targets)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(c_matrix) / c_matrix.sum(axis=1)

    if np.any(np.isnan(per_class)):
        per_class = per_class[~np.isnan(per_class)]

    bal_acc = np.mean(per_class)

    return {
        "acc": correct / correct_total,
        "bal_acc": bal_acc,
    }
