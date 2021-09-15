"""
Created on 2021/01/05
@author Sangwoo Han
"""

import warnings
from functools import partial
from typing import Dict, Hashable, Iterable, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

TPredict = np.ndarray
TTarget = Iterable[Iterable[Hashable]]
TMlb = Optional[MultiLabelBinarizer]


def get_accuracy(labels: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
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
        "per_class": per_class,
    }


def get_precision_results(
    prediction: TPredict,
    targets: TTarget,
    inv_w: Optional[np.ndarray] = None,
    mlb: TMlb = None,
) -> Dict[str, float]:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(targets)

    prediction = mlb.classes_[prediction]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        p1 = get_p_1(prediction, targets, mlb)
        p3 = get_p_3(prediction, targets, mlb)
        p5 = get_p_5(prediction, targets, mlb)

        n1 = get_n_1(prediction, targets, mlb)
        n3 = get_n_3(prediction, targets, mlb)
        n5 = get_n_5(prediction, targets, mlb)

        r1 = get_r_1(prediction, targets, mlb)
        r5 = get_r_5(prediction, targets, mlb)
        r10 = get_r_10(prediction, targets, mlb)

        ret = {
            "p1": p1,
            "p3": p3,
            "p5": p5,
            "n1": n1,
            "n3": n3,
            "n5": n5,
            "r1": r1,
            "r5": r5,
            "r10": r10,
        }

        if inv_w is not None:
            psp1 = get_psp_1(prediction, targets, inv_w, mlb)
            psp3 = get_psp_3(prediction, targets, inv_w, mlb)
            psp5 = get_psp_5(prediction, targets, inv_w, mlb)

            psn1 = get_psndcg_1(prediction, targets, inv_w, mlb)
            psn3 = get_psndcg_3(prediction, targets, inv_w, mlb)
            psn5 = get_psndcg_5(prediction, targets, inv_w, mlb)

            ret = {
                **ret,
                "psp1": psp1,
                "psp3": psp3,
                "psp5": psp5,
                "psn1": psn1,
                "psn3": psn3,
                "psn5": psn5,
            }

    return ret


def get_precision_results2(
    prediction: TPredict,
    targets: TTarget,
    inv_w: Optional[np.ndarray] = None,
    mlb: TMlb = None,
) -> Dict[str, float]:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(targets)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        p1 = get_p_1(prediction, targets, mlb)
        p3 = get_p_3(prediction, targets, mlb)
        p5 = get_p_5(prediction, targets, mlb)
        p10 = get_p_10(prediction, targets, mlb)

        n1 = get_n_1(prediction, targets, mlb)
        n3 = get_n_3(prediction, targets, mlb)
        n5 = get_n_5(prediction, targets, mlb)
        n10 = get_n_10(prediction, targets, mlb)

        r1 = get_r_1(prediction, targets, mlb)
        r5 = get_r_5(prediction, targets, mlb)
        r10 = get_r_10(prediction, targets, mlb)

        ret = {
            "p1": p1,
            "p3": p3,
            "p5": p5,
            "p10": p10,
            "n1": n1,
            "n3": n3,
            "n5": n5,
            "n10": n10,
            "r1": r1,
            "r5": r5,
            "r10": r10,
        }

        if inv_w is not None:
            psp1 = get_psp_1(prediction, targets, inv_w, mlb)
            psp3 = get_psp_3(prediction, targets, inv_w, mlb)
            psp5 = get_psp_5(prediction, targets, inv_w, mlb)
            psp10 = get_psp_10(prediction, targets, inv_w, mlb)
            psp20 = get_psp_20(prediction, targets, inv_w, mlb)

            psn1 = get_psndcg_1(prediction, targets, inv_w, mlb)
            psn3 = get_psndcg_3(prediction, targets, inv_w, mlb)
            psn5 = get_psndcg_5(prediction, targets, inv_w, mlb)
            psn10 = get_psndcg_10(prediction, targets, inv_w, mlb)
            psn20 = get_psndcg_20(prediction, targets, inv_w, mlb)

            ret = {
                **ret,
                "psp1": psp1,
                "psp3": psp3,
                "psp5": psp5,
                "psp10": psp10,
                "psp20": psp20,
                "psn1": psn1,
                "psn3": psn3,
                "psn5": psn5,
                "psn10": psn10,
                "psn20": psn20,
            }

    return ret


def get_precision(
    prediction: TPredict,
    targets: TTarget,
    mlb: TMlb = None,
    top=5,
) -> float:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(targets)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top])
    return prediction.multiply(targets).sum() / (top * targets.shape[0])


get_p_1 = partial(get_precision, top=1)
get_p_3 = partial(get_precision, top=3)
get_p_5 = partial(get_precision, top=5)
get_p_10 = partial(get_precision, top=10)


def get_ndcg(prediction: TPredict, targets: TTarget, mlb: TMlb = None, top=5) -> float:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(targets)
    log = 1.0 / np.log2(np.arange(top) + 2)
    dcg = np.zeros((targets.shape[0], 1))
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    for i in range(top):
        p = mlb.transform(prediction[:, i : i + 1])
        dcg += p.multiply(targets).sum(axis=-1) * log[i]
    return np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top) - 1])


get_n_1 = partial(get_ndcg, top=1)
get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)
get_n_10 = partial(get_ndcg, top=10)


def get_inv_propensity(train_y: csr_matrix, a=0.55, b=1.5) -> np.ndarray:
    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)


def get_psp(
    prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None, top=5
) -> float:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(targets)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top]).multiply(inv_w)
    num = prediction.multiply(targets).sum()
    t, den = csr_matrix(targets.multiply(inv_w)), 0
    for i in range(t.shape[0]):
        den += np.sum(np.sort(t.getrow(i).data)[-top:])
    return num / den


get_psp_1 = partial(get_psp, top=1)
get_psp_3 = partial(get_psp, top=3)
get_psp_5 = partial(get_psp, top=5)
get_psp_10 = partial(get_psp, top=10)
get_psp_20 = partial(get_psp, top=20)


def get_psndcg(
    prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None, top=5
) -> float:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(targets)
    log = 1.0 / np.log2(np.arange(top) + 2)
    psdcg = 0.0
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    for i in range(top):
        p = mlb.transform(prediction[:, i : i + 1]).multiply(inv_w)
        psdcg += p.multiply(targets).sum() * log[i]
    t, den = csr_matrix(targets.multiply(inv_w)), 0.0
    for i in range(t.shape[0]):
        num = min(top, len(t.getrow(i).data))
        den += -np.sum(np.sort(-t.getrow(i).data)[:num] * log[:num])
    return psdcg / den


get_psndcg_1 = partial(get_psndcg, top=1)
get_psndcg_3 = partial(get_psndcg, top=3)
get_psndcg_5 = partial(get_psndcg, top=5)
get_psndcg_10 = partial(get_psndcg, top=10)
get_psndcg_20 = partial(get_psndcg, top=20)


def get_recall(
    prediction: TPredict,
    targets: TTarget,
    mlb: TMlb = None,
    top=5,
) -> float:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(targets)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top])
    return (prediction.multiply(targets).sum(axis=-1) / targets.sum(axis=-1)).mean()


get_r_1 = partial(get_recall, top=1)
get_r_3 = partial(get_recall, top=3)
get_r_5 = partial(get_recall, top=5)
get_r_10 = partial(get_recall, top=10)
