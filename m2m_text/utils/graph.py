"""
Created on 2021/04/05
@author Sangwoo Han
"""
from typing import Dict, List, Union

import numpy as np
import torch
from dgl.dataloading import distributed, sampling, subgraph
from dgl.dataloading.neighbor import BlockSampler
from dgl.heterograph import DGLBlock
from scipy.sparse import csgraph

from ..datasets._base import Dataset


def get_ease_weight(dataset: Dataset, lamda: float) -> np.ndarray:
    """Get B matrix using EASE algorithm.

    Paper: https://arxiv.org/pdf/1905.03375.pdf

    Args:
        dataset (Dataset): `Dataset` instance.
        lamda (float): L2-norm regularization parameter.

    Returns:
        B (np.ndarray): LxL weight matrix.
    """
    G = (dataset.y.T @ dataset.y).astype(np.float64)
    diag_indices = np.diag_indices(G.shape[0])
    G[diag_indices] += lamda
    P = np.linalg.inv(G.toarray())
    B = P / (-np.diag(P))
    B[diag_indices] = 0

    return B


def get_adj(
    b: np.ndarray,
    top_adj: Union[int, float],
    use_b_weights: bool = False,
    laplacian_norm: bool = True,
) -> np.ndarray:
    """Get adjacency matrix from B matrix.

    Args:
        b (np.ndarray): B matrix from EASE.

        top_adj (int or float): If the type of top_adj is int,
            top-k weights are remained per row. If the type of top_adj is float,
            the weights which is greater or equal to top_adj are remained.

         laplacian_norm (bool): Apply laplacian norm when true.

    Returns:
        adj (np.ndarray): Adjacency matrix.
    """
    adj = np.zeros_like(b, dtype=np.float32)

    if type(top_adj) == int:
        indices = np.argsort(b)[:, ::-1][:, :top_adj]

        rows = np.array(
            [[i for _ in range(top_adj)] for i in range(b.shape[0])]
        ).reshape(-1)
        cols = indices.reshape(-1)

        indices = (rows, cols)
    else:
        indices = np.where(b >= top_adj)

    if use_b_weights:
        adj[indices] = b[indices]
    else:
        adj[indices] = 1

    adj = csgraph.laplacian(adj, normed=laplacian_norm)

    return adj


def get_random_adj(
    dim: int,
    sparsity: float,
    laplacian_norm: bool = True,
) -> np.ndarray:
    """Get random adjacency matrix.

    Args:
        dim (int): Dimension of adjacency matrix (dim x dim).
        sparsity (float): Sparsity of matrix (proportion of 0).
        laplacian_norm (bool): Apply laplacian norm when true.

    Returns:
        adj (np.ndarray): Generated adjacency matrix.
    """
    adj = np.zeros((dim, dim))
    triu_inds = np.triu_indices(dim, 1)
    mask = np.random.binomial(1, 1 - sparsity, triu_inds[0].shape[0]).astype(np.bool)
    value = np.random.randn(mask.shape[0])
    adj[triu_inds[0][mask], triu_inds[1][mask]] = value[mask]
    adj.T[triu_inds[0][mask], triu_inds[1][mask]] = value[mask]

    adj = csgraph.laplacian(adj, normed=laplacian_norm)

    return adj


def stack_embed(
    blocks: DGLBlock,
    nids: List[int],
    h_embeds: Union[Dict[str, List[torch.Tensor]], List[torch.Tensor]],
    node_type: str = None,
) -> torch.Tensor:
    h_embed = []

    for h, block in zip(h_embeds[:-1], blocks[1:]):
        temp = []
        for _id in nids:
            if node_type:
                index = torch.where(block.srcdata["_ID"][node_type] == _id)[0]
            else:
                index = torch.where(block.srcdata["_ID"] == _id)[0]

            assert index.nelement() != 0

            if node_type:
                temp.append(h[node_type][index])
            else:
                temp.append(h[index])

        h_embed.append(torch.cat(temp))

    if node_type:
        h_embed.append(h_embeds[-1][node_type])
    else:
        h_embed.append(h_embeds[-1])

    return torch.stack(h_embed, dim=1)


class MultiLayerNeighborSampler(BlockSampler):
    """
    Reference: https://docs.dgl.ai/en/0.6.x/_modules/dgl/dataloading/neighbor.html#MultiLayerNeighborSampler
    """

    def __init__(self, fanouts, replace=False, prob=None, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if isinstance(g, distributed.DistGraph):
            if fanout is None:
                # TODO(zhengda) There is a bug in the distributed version of in_subgraph.
                # let's use sample_neighbors to replace in_subgraph for now.
                frontier = distributed.sample_neighbors(
                    g, seed_nodes, -1, replace=False
                )
            else:
                frontier = distributed.sample_neighbors(
                    g, seed_nodes, fanout, replace=self.replace
                )
        else:
            if fanout is None:
                frontier = subgraph.in_subgraph(g, seed_nodes)
            else:
                frontier = sampling.sample_neighbors(
                    g, seed_nodes, fanout, replace=self.replace, prob=self.prob
                )
        return frontier
