"""
Created on 2021/01/07
@author Sangwoo Han
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def classwise_loss(outputs, targets, multi_label=False):
    """
    Reference: https://github.com/alinlab/M2m/blob/master/utils.py
    """
    if multi_label:
        return (outputs * targets).mean()

    out_1hot = torch.ones_like(outputs)
    out_1hot.scatter_(1, targets.view(-1, 1), -1)
    return (outputs * out_1hot).mean()


class CosineDistance(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return 1 - F.cosine_similarity(x1, x2, self.dim, self.eps)


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1, eps=1e-8, loss_fct=nn.MSELoss()):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.loss_fct = loss_fct

    def forward(self, x1, x2, labels):
        return self.loss_fct(
            F.cosine_similarity(x1, x2, self.dim, self.eps), labels.view(-1)
        )


class CircleLoss(nn.Module):
    """Implementaion of Circle loss

    Paper: https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Circle_Loss_A_Unified_Perspective_of_Pair_Similarity_Optimization_CVPR_2020_paper.pdf
    """

    def __init__(
        self,
        m: float = 0.15,
        gamma: float = 1.0,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        if distance_function is None:
            self.distance_function = nn.CosineSimilarity()

    def forward(
        self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
    ) -> torch.Tensor:
        sp = self.distance_function(anchor, pos)
        sn = self.distance_function(anchor, neg)

        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0)
        )

        return loss


class CircleLoss2(nn.Module):
    """Implementaion of Circle loss supporting n-pairs"""

    def __init__(
        self,
        m: float = 0.15,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.m = m
        self.gamma = gamma

    def forward(
        self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
    ) -> torch.Tensor:
        anchor = F.normalize(anchor, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        sp = (anchor.unsqueeze(1) @ pos.transpose(2, 1)).squeeze()
        sn = (anchor.unsqueeze(1) @ neg.transpose(2, 1)).squeeze()

        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = (
            F.softplus(torch.logsumexp(logit_p, dim=-1)).mean()
            + F.softplus(torch.logsumexp(logit_n, dim=-1)).mean()
        )

        return loss


class CircleLoss3(nn.Module):
    """Implementaion of Circle loss"""

    def __init__(
        self,
        m: float = 0.15,
        gamma: float = 1.0,
        metric: str = "cosine",
    ) -> None:
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.metric = metric

        assert self.metric in ["cosine", "euclidean"]

    def distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.metric == "cosine":
            x1 = F.normalize(x1, dim=-1)
            x2 = F.normalize(x2, dim=-1)
            return (x1.unsqueeze(1) @ x2.transpose(2, 1)).squeeze()
        return 1 / (1 + torch.cdist(x1.unsqueeze(1), x2).squeeze())

    def forward(
        self,
        anchor: torch.Tensor,
        pos: torch.Tensor,
        neg: Optional[torch.Tensor] = None,
        pos_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        sp = self.distance(anchor, pos)
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.0)
        delta_p = 1 - self.m
        weights = 1.0 if pos_weights is None else pos_weights
        logit_p = -ap * (sp - delta_p) * self.gamma * weights
        logit_p_logsumexp = torch.logsumexp(logit_p, dim=-1)

        if neg is not None:
            sn = self.distance(anchor, neg)
            an = torch.clamp_min(sn.detach() + self.m, min=0.0)
            delta_n = self.m
            neg_weights = 1.0 if pos_weights is None else pos_weights.mean()
            logit_n = an * (sn - delta_n) * self.gamma * neg_weights
            logit_n_logsumexp = torch.logsumexp(logit_n, dim=-1)
        else:
            logit_n_logsumexp = 0

        loss = F.softplus(logit_p_logsumexp + logit_n_logsumexp)

        return loss.mean()
