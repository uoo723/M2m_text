import os
from contextlib import redirect_stderr
from functools import reduce
from typing import Iterable, List, Optional, Union

import numpy as np
from logzero import logger
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from .utils.data import get_sparse_features

__all__ = ["get_clusters"]


def get_clusters(
    labels_f: csr_matrix,
    levels: List[int] = [10],
    eps: float = 1e-4,
    max_leaf: int = 2,
    verbose: bool = False,
) -> List[np.ndarray]:

    if verbose:
        logger.info("Clustering")
        logger.info(f"Start Clustering {levels}")

    levels, q = [2 ** x for x in levels], None

    clusters = []

    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]

    while q:
        labels_list = np.asarray([x[0] for x in q], dtype=np.object)

        assert (
            len(reduce(lambda a, b: a | set(b), labels_list, set()))
            == labels_f.shape[0]
        )

        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            clusters.append(np.asarray(labels_list, dtype=np.object))

            if verbose:
                logger.info(f"Finish Clustering Level-{level}")

            if level == len(levels) - 1:
                break

        else:
            if verbose:
                logger.info(f"Finish Clustering {len(labels_list)}")

        next_q = []

        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))

        q = next_q

    if verbose:
        logger.info("Finish Clustering")

    return clusters


def build_tree_by_level(
    sparse_features_path: str,
    labels: Iterable[Iterable[str]],
    mlb: MultiLabelBinarizer,
    groups_path: str,
    levels: list = [10],
    eps: float = 1e-4,
    max_leaf: int = 2,
    tokenized_texts: Optional[Union[Iterable[Iterable[str]], Iterable[str]]] = None,
    indices: np.ndarray = None,
):
    os.makedirs(os.path.split(groups_path)[0], exist_ok=True)
    logger.info("Clustering")
    logger.info("Getting Labels Feature")

    sparse_x = get_sparse_features(sparse_features_path, tokenized_texts)

    with redirect_stderr(None):
        sparse_y = mlb.transform(labels)

    if indices is not None:
        sparse_x = sparse_x[indices]
        sparse_y = sparse_y[indices]

    labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))

    logger.info(f"Start Clustering {levels}")

    levels, q = [2 ** x for x in levels], None

    for i in range(len(levels) - 1, -1, -1):
        if os.path.exists(f"{groups_path}-Level-{i}.npy"):
            labels_list = np.load(f"{groups_path}-Level-{i}.npy", allow_pickle=True)
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break

    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]

    while q:
        labels_list = np.asarray([x[0] for x in q], dtype=np.object)

        assert (
            len(reduce(lambda a, b: a | set(b), labels_list, set()))
            == labels_f.shape[0]
        )

        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            groups = np.asarray(labels_list, dtype=np.object)
            logger.info(f"Finish Clustering Level-{level}")

            np.save(f"{groups_path}-Level-{level}.npy", groups)

            if level == len(levels) - 1:
                break

        else:
            logger.info(f"Finish Clustering {len(labels_list)}")

        next_q = []

        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))

        q = next_q

    logger.info("Finish Clustering")


def split_node(
    labels_i: np.ndarray,
    labels_f: csr_matrix,
    eps: float,
    alg: str = "kmeans",
    overlap_ratio: float = 0.0,
    return_centers: bool = False,
):
    n = len(labels_i)
    n_overlap = int(n // 2 * overlap_ratio)
    centers = None

    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    old_dis, new_dis = -10000.0, -1.0

    if type(labels_f) == csr_matrix:
        centers = labels_f[[c1, c2]].toarray()
    else:
        centers = labels_f[[c1, c2]]

    l_labels_i, r_labels_i = None, None

    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = (
            partition[: n // 2 + n_overlap],
            partition[n // 2 - n_overlap :],
        )
        old_dis, new_dis = (
            new_dis,
            (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n,
        )
        centers = normalize(
            np.asarray(
                [
                    np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                    np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0))),
                ]
            )
        )

    ret = (labels_i[l_labels_i], labels_f[l_labels_i]), (
        labels_i[r_labels_i],
        labels_f[r_labels_i],
    )

    if return_centers and centers is not None:
        ret += (centers,)

    return ret
