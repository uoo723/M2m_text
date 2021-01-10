"""
Created on 2021/01/07
@author Sangwoo Han
"""
import random
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    ckpt_path: str,
    model: Union[nn.DataParallel, nn.Module],
    optim: Optimizer,
    results: dict,
    epoch: int,
    scheduler: Optional[_LRScheduler] = None,
    other_states: dict = {},
):
    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        "net": model.state_dict(),
        "optimizer": optim.state_dict(),
        "results": results,
        "epoch": epoch,
        "rng_state": (torch.get_rng_state(), np.random.get_state(), random.getstate()),
        **({} if scheduler is None else {"scheduler": scheduler.state_dict()}),
        "other_states": other_states,
    }

    torch.save(state, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    model: Optional[Union[nn.DataParallel, nn.Module]] = None,
    optim: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    set_rng_state: bool = True,
    return_other_states: bool = False,
    **torch_load_args,
) -> int:
    ckpt = torch.load(ckpt_path, **torch_load_args)

    if model is not None:
        if isinstance(model, nn.DataParallel):
            model = model.module

        model.load_state_dict(ckpt["net"])

    if optim:
        optim.load_state_dict(ckpt["optimizer"])

    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])

    if set_rng_state:
        torch.set_rng_state(ckpt["rng_state"][0])
        np.random.set_state(ckpt["rng_state"][1])
        random.setstate(ckpt["rng_state"][2])

    if return_other_states:
        ret = (ckpt["epoch"], ckpt["other_states"])

    else:
        ret = ckpt["epoch"]

    return ret
