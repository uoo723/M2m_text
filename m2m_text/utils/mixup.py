"""
Created on 2021/02/14
@author Sangwoo Han
"""
from typing import Iterable, Optional, Tuple, Union

import torch


def mixup(x1: torch.Tensor, x2: torch.Tensor, lamda: int) -> torch.Tensor:
    return lamda * x1 + (1 - lamda) * x2


class MixUp:
    def __init__(
        self,
        alpha: float = 0.4,
        num_labels: Optional[float] = None,
        n_samples_per_class: Optional[torch.Tensor] = None,
    ) -> None:
        self.m = torch.distributions.Beta(alpha, alpha)
        self.num_labels = num_labels
        self.n_samples_per_class = n_samples_per_class

    def __call__(
        self, train_x: torch.Tensor, train_y: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        if self.num_labels is not None:
            lamda = self.m.sample((self.num_labels,)).to(train_x.device)
        else:
            lamda = self.m.sample()

        indices = torch.randperm(train_x.size(0))

        if self.num_labels is not None and len(train_x.shape) == 3:
            lamda_x = lamda.unsqueeze(-1)
        else:
            lamda_x = lamda

        if self.n_samples_per_class is not None:
            rows, cols = train_y.nonzero(as_tuple=True)
            probs = self.n_samples_per_class[cols] / torch.max(self.n_samples_per_class)
            mask_idx = torch.bernoulli(probs)
            mask = torch.zeros(train_y.shape)
            mask[rows, cols] = mask_idx
            lamda = torch.where(mask > lamda, mask, lamda)

        mixed_x = mixup(train_x, train_x[indices], lamda_x)
        ret = mixed_x

        if train_y is not None:
            mixed_y = mixup(train_y, train_y[indices], lamda)
            ret = (ret, mixed_y)

        return ret
