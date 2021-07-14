"""
Created on 2021/01/07
@author Sangwoo Han
"""
import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

TModel = Union[nn.DataParallel, nn.Module]


def save_checkpoint(
    ckpt_path: str,
    model: TModel,
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
    model: Optional[TModel] = None,
    optim: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    set_rng_state: bool = True,
    return_other_states: bool = False,
    **torch_load_args,
) -> int:
    print(f"ckpt:{ckpt_path}")
    ckpt = torch.load(ckpt_path, **torch_load_args)

    if model is not None and "net" in ckpt:
        if isinstance(model, nn.DataParallel):
            model = model.module

        model.load_state_dict(ckpt["net"])

    if optim and "optimizer" in ckpt:
        optim.load_state_dict(ckpt["optimizer"])

    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if set_rng_state and "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"][0])
        np.random.set_state(ckpt["rng_state"][1])
        random.setstate(ckpt["rng_state"][2])

    if return_other_states:
        ret = (ckpt["epoch"], ckpt.get("other_states", {}))

    else:
        ret = ckpt["epoch"]

    return ret


def copy_model_parameters(from_model: TModel, to_model: TModel) -> None:
    if isinstance(from_model, nn.DataParallel):
        from_model = from_model.module

    if isinstance(to_model, nn.DataParallel):
        to_model = to_model.module

    to_model.load_state_dict(from_model.state_dict())


def get_model_outputs(
    model: TModel,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor]],
    other_inputs: Optional[Tuple[torch.Tensor]] = (),
    input_opts: Optional[Dict] = {},
    is_transformer: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    if is_transformer:
        if isinstance(inputs, (list, tuple)):
            model_inputs = {"input_ids": inputs[0], "attention_mask": inputs[1]}
        else:
            model_inputs = {"outputs": (inputs, *other_inputs)}
    else:
        if other_inputs:
            model_inputs = (inputs, *other_inputs)
        else:
            model_inputs = inputs

    if is_transformer:
        outputs = model(**model_inputs, **input_opts)[0]
        if type(inputs) == tuple:
            outputs = (outputs, inputs[1])
    else:
        outputs = model(model_inputs, **input_opts)

    return outputs
