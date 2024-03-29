"""
Created on 2021/05/13
@author Sangwoo Han
"""

# Refrence: https://github.com/VarIr/scikit-hubness/tree/master/skhubness/neighbors

import shutil
from abc import ABC, abstractmethod
from multiprocessing import Process, cpu_count
from typing import Optional, Tuple, Union

import nmslib
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted


def check_n_candidates(n_candidates):
    # Check the n_neighbors parameter
    if n_candidates <= 0:
        raise ValueError(f"Expected n_neighbors > 0. Got {n_candidates:d}")
    if not np.issubdtype(type(n_candidates), np.integer):
        raise TypeError(
            f"n_neighbors does not take {type(n_candidates)} value, enter integer value"
        )
    return n_candidates


class ApproximateNearestNeighbor(ABC):
    """Abstract base class for approximate nearest neighbor search methods.
    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are "angular", "euclidean", "manhattan", "hamming", "dot"
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.
    """

    def __init__(
        self,
        n_candidates: int = 5,
        metric: str = "sqeuclidean",
        n_jobs: int = 1,
        verbose: int = 0,
        *args,
        **kwargs,
    ):
        self.n_candidates = n_candidates
        self.metric = metric
        if n_jobs is None:
            n_jobs = 1
        elif n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs
        self.verbose = verbose

    @abstractmethod
    def fit(self, X, y=None):
        """Setup ANN index from training data.
        Parameters
        ----------
        X: np.array
            Data to be indexed
        y: any
            Ignored
        """
        pass  # pragma: no cover

    @abstractmethod
    def kneighbors(
        self, X=None, n_candidates=None, return_distance=True
    ) -> Union[Tuple[np.array, np.array], np.array]:
        """Retrieve k nearest neighbors.
        Parameters
        ----------
        X: np.array or None, optional, default = None
            Query objects. If None, search among the indexed objects.
        n_candidates: int or None, optional, default = None
            Number of neighbors to retrieve.
            If None, use the value passed during construction.
        return_distance: bool, default = True
            If return_distance, will return distances and indices to neighbors.
            Else, only return the indices.
        """
        pass  # pragma: no cover


class HNSW(ApproximateNearestNeighbor):
    """Wrapper for using nmslib
    Hierarchical navigable small-world graphs are data structures,
    that allow for approximate nearest neighbor search.
    Here, an implementation from nmslib is used.
    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are "angular", "euclidean", "manhattan", "hamming", "dot"
    method: str, default = 'hnsw',
        ANN method to use. Currently, only 'hnsw' is supported.
    post_processing: int, default = 2
        More post processing means longer index creation,
        and higher retrieval accuracy.
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose >= 2, show progress bar on indexing.
    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures
    """

    valid_metrics = [
        "euclidean",
        "l2",
        "minkowski",
        "squared_euclidean",
        "sqeuclidean",
        "cosine",
        "cosinesimil",
    ]

    def __init__(
        self,
        M: int = 64,
        efC: int = 300,
        efS: int = 300,
        n_candidates: int = 5,
        metric: str = "euclidean",
        method: str = "hnsw",
        post_processing: int = 2,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        if nmslib is None:  # pragma: no cover
            raise ImportError(
                f"Please install the `nmslib` package, before using this class.\n"
                f"$ pip install nmslib"
            ) from None

        super().__init__(
            n_candidates=n_candidates, metric=metric, n_jobs=n_jobs, verbose=verbose
        )
        self.M = M
        self.efC = efC
        self.efS = efS
        self.method = method
        self.post_processing = post_processing
        self.space = None
        self._embeddings = None
        self._input_embeddings = None

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: np.ndarray) -> None:
        self._embeddings = value

    @property
    def input_embeddings(self) -> np.ndarray:
        return self._input_embeddings

    @input_embeddings.setter
    def input_embeddings(self, value: np.ndarray) -> None:
        self._input_embeddings = value

    def set_space(self):
        if self.metric in [
            "euclidean",
            "l2",
            "minkowski",
            "squared_euclidean",
            "sqeuclidean",
        ]:
            if self.metric in ["squared_euclidean", "sqeuclidean"]:
                self.metric = "sqeuclidean"
            else:
                self.metric = "euclidean"
            self.space = "l2"
        elif self.metric in ["cosine", "cosinesimil"]:
            self.space = "cosinesimil"
        else:
            raise ValueError(
                f'Invalid metric "{self.metric}". Please try "euclidean" or "cosine".'
            )

    def fit(self, X, y=None):
        """Setup the HNSW index from training data.
        Parameters
        ----------
        X: np.array
            Data to be indexed
        y: any
            Ignored
        Returns
        -------
        self: HNSW
            An instance of HNSW with a built graph
        """
        X = check_array(X)

        self._embeddings = X

        method = self.method
        post_processing = self.post_processing

        self.set_space()

        hnsw_index = nmslib.init(method=method, space=self.space)
        hnsw_index.addDataPointBatch(X)
        hnsw_index.createIndex(
            {
                "M": self.M,
                "post": post_processing,
                "indexThreadQty": self.n_jobs,
                "efConstruction": self.efC,
            },
            print_progress=(self.verbose >= 2),
        )
        self.index_ = hnsw_index
        self.n_samples_fit_ = len(self.index_)

        assert self.space in [
            "l2",
            "cosinesimil",
        ], f"Internal: self.space={self.space} not allowed"

        return self

    def kneighbors(
        self,
        X: np.ndarray = None,
        n_candidates: int = None,
        return_distance: bool = True,
        search_by_id: bool = False,
    ) -> Union[Tuple[np.array, np.array], np.array]:
        """Retrieve k nearest neighbors.
        Parameters
        ----------
        X: np.array or None, optional, default = None
            Query objects. If None, search among the indexed objects.
        n_candidates: int or None, optional, default = None
            Number of neighbors to retrieve.
            If None, use the value passed during construction.
        return_distance: bool, default = True
            If return_distance, will return distances and indices to neighbors.
            Else, only return the indices.
        """
        check_is_fitted(
            self,
            [
                "index_",
            ],
        )

        if X is None:
            raise NotImplementedError(f"Please provide X to hnsw.kneighbors().")

        # Check the n_neighbors parameter
        if n_candidates is None:
            n_candidates = self.n_candidates
        n_candidates = check_n_candidates(n_candidates)

        self.index_.setQueryTimeParams({"efSearch": self.efS})

        if search_by_id:
            X = self.embeddings[X]

        # Fetch the neighbor candidates
        neigh_ind_dist = self.index_.knnQueryBatch(
            X, k=n_candidates, num_threads=self.n_jobs
        )

        # If fewer candidates than required are found for a query,
        # we save index=-1 and distance=NaN
        n_test = X.shape[0]
        neigh_ind = -np.ones((n_test, n_candidates), dtype=np.int32)
        neigh_dist = np.empty_like(neigh_ind, dtype=X.dtype) * np.nan

        for i, (ind, dist) in enumerate(neigh_ind_dist):
            neigh_ind[i, : ind.size] = ind
            neigh_dist[i, : dist.size] = dist

        # Convert cosine similarities to cosine distances
        if self.space == "cosinesimil":
            neigh_dist *= -1
            neigh_dist += 1
        elif self.space == "l2" and self.metric == "sqeuclidean":
            neigh_dist **= 2

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind

    def save_index(self, filename: str, save_data: bool = True):
        check_is_fitted(
            self,
            [
                "index_",
            ],
        )

        self.index_.saveIndex(filename, save_data)

    def load_index(self, filename: str, load_data: bool = True):
        if not hasattr(self, "index_"):
            self.set_space()
            hnsw_index = nmslib.init(method=self.method, space=self.space)
            self.index_ = hnsw_index

        self.index_.loadIndex(filename, load_data)
        self.n_samples_fit_ = len(self.index_)


def build_ann(
    embeddings: Optional[np.ndarray] = None,
    M: int = 100,
    efC: int = 300,
    efS: int = 500,
    n_candidates: int = 500,
    metric: str = "cosine",
    n_jobs: int = -1,
    filepath: Optional[str] = None,
    embedding_filepath: Optional[str] = None,
) -> HNSW:
    index = HNSW(
        M=M, efC=efC, efS=efS, n_candidates=n_candidates, metric=metric, n_jobs=n_jobs
    )

    if embeddings is not None:
        index.fit(embeddings)

    if filepath is not None:
        assert embedding_filepath is not None

        index.save_index(filepath)
        np.save(embedding_filepath, embeddings)

    return index


def build_ann_async(
    filepath: str,
    embedding_filepath: str,
    embeddings: np.ndarray,
    M: int = 100,
    efC: int = 300,
    efS: int = 500,
    n_candidates: int = 500,
    metric: str = "cosine",
    n_jobs: int = -1,
) -> Process:
    p = Process(
        target=build_ann,
        args=(
            embeddings,
            M,
            efC,
            efS,
            n_candidates,
            metric,
            n_jobs,
            filepath,
            embedding_filepath,
        ),
    )

    p.start()

    return p


def load_ann(
    index: HNSW,
    filepath: str,
    embedding_filepath: str,
    p: Optional[Process] = None,
) -> bool:
    if p is not None:
        if p.is_alive():
            return False

        if p.exitcode != 0:
            raise Exception(f"Building process failed. exit code: {p.exitcode}")

    index.load_index(filepath)
    index.embeddings = np.load(embedding_filepath)

    return True


def copy_ann_index(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)
    shutil.copyfile(src + ".dat", dst + ".dat")
