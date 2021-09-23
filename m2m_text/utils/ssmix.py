"""
Created on 2021/09/20
@author Sangwoo Han
@reference: https://github.com/clovaai/ssmix/blob/master/trainer.py
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .train import clone_tensor

TModel = [nn.Module, nn.DataParallel]
TInput = Union[torch.Tensor, Dict[str, torch.Tensor]]
TTarget = torch.Tensor


def split_batch(
    batch_x: TInput, batch_y: TTarget
) -> Optional[Tuple[TInput, TTarget, TInput, TTarget]]:
    batch_size = (
        batch_x["input_ids"].size(0) if type(batch_x) == dict else batch_x.size(0)
    )

    if batch_size % 2 != 0:
        # Skip odd-numbered batch
        return None, None, None, None

    half_size = batch_size // 2

    if type(batch_x) == dict:
        inputs_left, inputs_right = {}, {}
        for key in batch_x.keys():
            inputs_left[key], inputs_right[key] = torch.split(batch_x[key], half_size)
    else:
        inputs_left, inputs_right = torch.split(batch_x, half_size)

    targets_left, targets_right = torch.split(batch_y, half_size)

    return inputs_left, targets_left, inputs_right, targets_right


def get_loss(
    model: TModel,
    criterion: nn.Module,
    inputs1: TInput,
    targets1: TTarget,
    targets2: TTarget = None,
    ratio: Optional[float] = None,
    **kwargs
) -> torch.Tensor:
    output = model(inputs1, **kwargs)[0]
    loss = criterion(output, targets1)
    if targets2 is not None:
        loss = loss * ratio + criterion(output, targets2) * (1 - ratio)
    loss = loss.mean()
    return loss


def get_saliency(
    model: TModel,
    criterion: nn.Module,
    inputs: TInput,
    targets: TTarget,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = targets.size(0)
    model.train()

    output = model(inputs, trace_grad=True, mp_enabled=False)

    logit: torch.Tensor = output[0]
    embeddings: torch.Tensor = output[-1]

    loss: torch.Tensor = criterion(logit, targets)

    loss.backward()

    unary = (embeddings.grad ** 2).mean(dim=-1).sqrt()
    unary = unary / unary.view(batch_size, -1).max(1)[0].view(batch_size, 1)

    model.zero_grad()

    return unary, embeddings


def ssmix_augment(
    model: TModel,
    criterion: nn.Module,
    inputs1: TInput,
    inputs2: TInput,
    targets1: TTarget,
    targets2: TTarget,
    ss_winsize: float,
) -> Tuple[TInput, torch.Tensor]:
    batch_size = targets1.size(0)
    inputs_aug = clone_tensor(inputs1)

    saliency1, _ = get_saliency(model, criterion, inputs1, targets1)
    saliency2, _ = get_saliency(model, criterion, inputs2, targets2)
    ratio = torch.ones((batch_size,), device=targets1.device)

    if type(inputs1) == dict:
        lengths1 = (inputs1["attention_mask"] != 0).sum(dim=-1)
        lengths2 = (inputs2["attention_mask"] != 0).sum(dim=-1)
        ignore_num_toks = 2
    else:
        lengths1 = (inputs1 != 0).sum(dim=-1)
        lengths2 = (inputs2 != 0).sum(dim=-1)
        ignore_num_toks = 0

    for i in range(batch_size):
        l1, l2 = lengths1[i].item(), lengths2[i].item()
        limit_len = l1 - ignore_num_toks
        mix_size = max(int(limit_len * ss_winsize), 1)

        if l2 < mix_size:
            ratio[i] = 1.0
            continue

        if l1 - ignore_num_toks == 0:
            continue

        saliency1_nopad = saliency1[i, :l1].unsqueeze(0).unsqueeze(0)
        saliency2_nopad = saliency2[i, :l2].unsqueeze(0).unsqueeze(0)

        saliency1_pool = (
            F.avg_pool1d(saliency1_nopad, mix_size, stride=1).squeeze(0).squeeze(0)
        )
        saliency2_pool = (
            F.avg_pool1d(saliency2_nopad, mix_size, stride=1).squeeze(0).squeeze(0)
        )

        if type(inputs1) == dict:
            saliency1_pool[0], saliency1_pool[-1] = 100, 100
            saliency2_pool[0], saliency2_pool[-1] = -100, -100
            input1_idx = saliency1_pool.argmin()
            input2_idx = saliency2_pool.argmax()
            inputs_aug["input_ids"][i, input1_idx : input1_idx + mix_size] = inputs2[
                "input_ids"
            ][i, input2_idx : input2_idx + mix_size]

            ratio[i] = 1 - (mix_size / (l1 - ignore_num_toks))
        else:
            input1_idx = saliency1_pool.argmin()
            input2_idx = saliency2_pool.argmax()
            inputs_aug[i, input1_idx : input1_idx + mix_size] = inputs2[
                i, input2_idx : input2_idx + mix_size
            ]

            ratio[i] = 1 - (mix_size / (l1 - ignore_num_toks))

    return inputs_aug, ratio
