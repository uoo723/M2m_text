"""
Created on 2021/02/14
@author Sangwoo Han
"""
from typing import Iterable, Optional, Tuple, Union

import torch


def mixup(x: torch.Tensor, lamda: int, indices: Iterable[int]) -> torch.Tensor:
    return lamda * x + (1 - lamda) * x[indices]


class MixUp:
    def __init__(self, alpha: float = 0.4) -> None:
        self.m = torch.distributions.Beta(alpha, alpha)

    def __call__(
        self, train_x: torch.Tensor, train_y: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        lamda = self.m.sample()
        indices = torch.randperm(train_x.size(0))
        mixed_x = mixup(train_x, lamda, indices)
        ret = mixed_x
        if train_y is not None:
            mixed_y = mixup(train_y, lamda, indices)
            ret = (ret, mixed_y)
        return ret
