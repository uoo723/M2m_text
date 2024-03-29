"""
Created on 2021/01/07
@author Sangwoo Han
"""
import os
import random
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

TModel = Union[nn.DataParallel, nn.Module]


def save_checkpoint(
    ckpt_path: str,
    model: TModel,
    epoch: int,
    label_encoder: Optional[TModel] = None,
    optim: Optional[Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[_LRScheduler] = None,
    results: Optional[dict] = None,
    other_states: dict = {},
) -> None:
    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {"net": model.state_dict()}
    state["epoch"] = epoch

    state["rng_state"] = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )

    if label_encoder is not None:
        if isinstance(label_encoder, nn.DataParallel):
            label_encoder = label_encoder.module
        state["le"] = label_encoder.state_dict()

    if optim is not None:
        state["optim"] = optim.state_dict()

    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    if results is not None:
        state["results"] = None

    state["other_states"] = other_states

    torch.save(state, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    model: Optional[TModel] = None,
    label_encoder: Optional[TModel] = None,
    optim: Optional[Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[_LRScheduler] = None,
    set_rng_state: bool = True,
    return_other_states: bool = False,
    **torch_load_args,
) -> int:
    ckpt = torch.load(ckpt_path, **torch_load_args)

    if model is not None and "net" in ckpt:
        if isinstance(model, nn.DataParallel):
            model = model.module

        model.load_state_dict(ckpt["net"])

    if label_encoder is not None and "le" in ckpt:
        if isinstance(label_encoder, nn.DataParallel):
            label_encoder = label_encoder.module

        label_encoder.load_state_dict(ckpt["le"])

    if optim and "optimizer" in ckpt:
        optim.load_state_dict(ckpt["optimizer"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

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


def save_checkpoint2(
    ckpt_path: str,
    epoch: int,
    models: Iterable[TModel],
    optim: Optional[Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[_LRScheduler] = None,
    results: Optional[Dict[str, float]] = None,
    other_states: Dict[str, Any] = {},
) -> None:

    states = {
        "models": [],
        "epoch": epoch,
        "rng_states": (torch.get_rng_state(), np.random.get_state(), random.getstate()),
        "other_states": other_states,
        "results": results,
    }

    for model in models:
        if isinstance(model, nn.DataParallel):
            model = model.module
        states["models"].append(model.state_dict())

    if optim is not None:
        states["optim"] = optim.state_dict()

    if scaler is not None:
        states["scaler"] = scaler.state_dict()

    if scheduler is not None:
        states["scheduler"] = scheduler.state_dict()

    torch.save(states, ckpt_path)


def load_checkpoint2(
    ckpt_path: str,
    models: Iterable[TModel],
    optim: Optional[Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[_LRScheduler] = None,
    set_rng_state: bool = True,
    return_other_states: bool = False,
) -> Union[int, Tuple[int, Dict[str, Any]]]:

    states = torch.load(ckpt_path)

    for i, model in enumerate(models):
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.load_state_dict(states["models"][i])

    if "optim" in states and optim is not None:
        optim.load_state_dict(states["optim"])

    if "scaler" in states and scaler is not None:
        scaler.load_state_dict(states["scaler"])

    if "scheduler" in states and scheduler is not None:
        scheduler.load_state_dict(states["scheduler"])

    if set_rng_state:
        torch.set_rng_state(states["rng_states"][0])
        np.random.set_state(states["rng_states"][1])
        random.setstate(states["rng_states"][2])

    if return_other_states:
        ret = (states["epoch"], states["other_states"])
    else:
        ret = states["epoch"]

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


def freeze_model(model: TModel, freeze: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(not freeze)
