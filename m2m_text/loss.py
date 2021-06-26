"""
Created on 2021/01/07
@author Sangwoo Han
"""
from typing import Callable

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
    def __init__(
        self,
        m: float = 0.25,
        gamma: float = 256,
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
