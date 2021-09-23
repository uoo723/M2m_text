"""
Created on 2021/07/06
@author Sangwoo Han

Instace Anchor new version, no cluster
"""
import copy
import os
import random
import re
import shutil
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch.nn as nn
from click_option_group import optgroup
from logzero import logger
from ruamel.yaml import YAML
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from m2m_text.datasets import (
    AmazonCat,
    AmazonCat13K,
    BertDataset,
    EURLex,
    EURLex4K,
    Wiki10,
    Wiki10_31K,
)
from m2m_text.datasets.sbert import collate_fn2
from m2m_text.datasets.text import TextDataset
from m2m_text.metrics import (
    get_inv_propensity,
    get_n_5,
    get_p_5,
    get_precision_results2,
    get_psp_5,
    get_r_10,
)
from m2m_text.networks import (
    AttentionRNN,
    AttentionRNN4Mix,
    LaCNN,
    LaRoberta,
    LaRoberta4Mix,
)
from m2m_text.optimizers import DenseSparseAdamW
from m2m_text.utils.data import copy_file, get_mlb
from m2m_text.utils.mixup import mixup
from m2m_text.utils.model import load_checkpoint2, save_checkpoint2
from m2m_text.utils.ssmix import get_loss, split_batch, ssmix_augment
from m2m_text.utils.train import (
    clip_gradient,
    log_elapsed_time,
    set_logger,
    set_seed,
    swa_init,
    swa_step,
    swap_swa_params,
    to_device,
)

DATASET_CLS = {
    "AmazonCat": AmazonCat,
    "AmazonCat13K": AmazonCat13K,
    "EURLex": EURLex,
    "EURLex4K": EURLex4K,
    "Wiki10": Wiki10,
    "Wiki10_31K": Wiki10_31K,
}

MODEL_CLS = {
    "AttentionRNN": AttentionRNN,
    "AttentionRNN4Mix": AttentionRNN4Mix,
    "LaRoberta": LaRoberta,
    "LaRoberta4Mix": LaRoberta4Mix,
    "LaCNN": LaCNN,
}

TRANSFORMER_MODELS = ["LaRoberta", "LaRoberta4Mix"]
MIX_MODELS = ["AttentionRNN4Mix", "LaRoberta4Mix"]


def get_model(
    model_cnf: dict,
    data_cnf: dict,
    num_labels: int,
    mp_enabled: bool,
    device: torch.device,
) -> nn.Module:
    model_cnf = copy.deepcopy(model_cnf)
    model_name = model_cnf["name"]

    if model_name in TRANSFORMER_MODELS:
        model = (
            MODEL_CLS[model_name]
            .from_pretrained(
                model_cnf["model"].pop("model_name"),
                num_labels=num_labels,
                mp_enabled=mp_enabled,
                **model_cnf["model"],
            )
            .to(device)
        )
    else:
        model = MODEL_CLS[model_name](
            num_labels=num_labels,
            mp_enabled=mp_enabled,
            **model_cnf["model"],
            **data_cnf["model"],
        ).to(device)

    return model


def get_dataset(
    model_cnf: Dict[str, Any], data_cnf: Dict[str, Any]
) -> Tuple[TextDataset, TextDataset, np.ndarray, np.ndarray]:
    dataset_name = data_cnf["name"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_dataset = DATASET_CLS[dataset_name](
            **data_cnf["dataset"], **model_cnf.get("dataset", {})
        )
        test_dataset = DATASET_CLS[dataset_name](
            train=False, **data_cnf["dataset"], **model_cnf.get("dataset", {})
        )

    train_ids = np.arange(len(train_dataset))
    train_ids, valid_ids = train_test_split(
        train_ids, test_size=data_cnf.get("valid_size", 200)
    )

    return train_dataset, test_dataset, train_ids, valid_ids


def get_dataloader(
    model_cnf: Dict[str, Any],
    train_dataset: TextDataset,
    test_dataset: TextDataset,
    train_ids: np.ndarray,
    valid_ids: np.ndarray,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int = 0,
    no_cuda: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    model_name = model_cnf["name"]

    if model_name in TRANSFORMER_MODELS:
        train_dataset = BertDataset(
            train_dataset, model_cnf["model"]["model_name"], **model_cnf["tokenizer"]
        )
        test_dataset = BertDataset(
            test_dataset, model_cnf["model"]["model_name"], **model_cnf["tokenizer"]
        )

        train_dataloader = DataLoader(
            Subset(train_dataset, train_ids),
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn2,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        valid_dataloader = DataLoader(
            Subset(train_dataset, valid_ids),
            batch_size=test_batch_size,
            collate_fn=collate_fn2,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            collate_fn=collate_fn2,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
    else:
        train_dataloader = DataLoader(
            Subset(train_dataset, train_ids),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        valid_dataloader = DataLoader(
            Subset(train_dataset, valid_ids),
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )

    return train_dataloader, valid_dataloader, test_dataloader


def set_lr(optim: Optimizer, lr: float, decay: float = None) -> None:
    for p in optim.param_groups:
        p["lr"] = lr
        if decay is not None and ["weight_decay"] != 0:
            p["weight_decay"] = decay


def train_step(
    global_step: int,
    accumulation_step: int,
    model: nn.Module,
    criterion: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_y: torch.Tensor,
    scaler: GradScaler,
    optim: Optimizer,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
):
    model.train()

    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        outputs = model(to_device(batch_x, device))[0]
        loss = criterion(outputs, to_device(batch_y, device))
        loss = loss / accumulation_step

    optim_start = time.time()

    loss = scaler.scale(loss)
    loss.backward()

    if (global_step + 1) % accumulation_step == 0:
        scaler.unscale_(optim)
        clip_gradient(model, gradient_norm_queue, gradient_clip_value)

        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

    optim_end = time.time()

    # print(f"optim: {(optim_end - optim_start) * 1000:.2f} ms")

    return loss.item()


def train_mixup_step(
    global_step: int,
    accumulation_step: int,
    model: nn.Module,
    criterion: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_y: torch.Tensor,
    scaler: GradScaler,
    optim: Optimizer,
    mixup_num: int,
    mixup_alpha: float,
    flow_mixup_enabled: bool,
    flow_alpha: float,
    no_label_smoothing: bool,
    input_opts: Dict[str, Any],
    output_opts: Dict[str, Any],
    inv_w: np.ndarray,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
):
    """random하게 두 샘플 mixup"""
    model.train()

    batch_x = to_device(batch_x, device)
    batch_y = to_device(batch_y, device)

    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        outputs = model(batch_x, **input_opts)
        outputs, others = outputs[0], outputs[1:]
        mixed_outputs = outputs.clone()
        mixed_batch_y = batch_y.clone()

        ridx = torch.randperm(batch_x.size(0))

        for i in range(batch_x.size(0)):
            y1_label_ids = batch_y[i].nonzero(as_tuple=True)[0]
            y2_label_ids = batch_y[ridx[i]].nonzero(as_tuple=True)[0]

            while True:
                y1_idx = (
                    torch.tensor(1 - 1 / inv_w[y1_label_ids.cpu()]).bernoulli().bool()
                )
                if len(y1_idx.size()) == 0:
                    y1_idx = y1_idx.unsqueeze(0)

                if y1_idx.sum() > 0:
                    break

            while True:
                y2_idx = (
                    torch.tensor(1 - 1 / inv_w[y2_label_ids.cpu()]).bernoulli().bool()
                )

                if len(y2_idx.size()) == 0:
                    y2_idx = y2_idx.unsqueeze(0)

                if y2_idx.sum() > 0:
                    break

            n = min(mixup_num, y2_idx.sum().item() + 1)
            lamda = (
                torch.distributions.Dirichlet(torch.tensor([mixup_alpha] * n))
                .sample((y1_idx.sum().item(),))
                .to(device)
                .to(outputs.dtype)
            )

            y2_idx2 = torch.multinomial(
                torch.ones(y2_idx.sum()).repeat(y1_idx.sum(), 1), n - 1
            )

            # print(
            #     "mixed_outputs[i, y1_label_ids[y1_idx]].shape:",
            #     mixed_outputs[i, y1_label_ids[y1_idx]].shape,
            # )
            # print(
            #     " mixed_outputs[ridx[i], y2_label_ids[y2_idx][y2_idx2]].shape",
            #     mixed_outputs[ridx[i], y2_label_ids[y2_idx][y2_idx2]].shape,
            # )
            # print("y1_idx.shape:", y1_idx.shape)
            # print("y1_idx.sum()", y1_idx.sum())
            # print("y1_label_Ids:", y1_label_ids)
            # print()
            mixed = torch.cat(
                [
                    mixed_outputs[i, y1_label_ids[y1_idx]].unsqueeze(1),
                    mixed_outputs[ridx[i], y2_label_ids[y2_idx][y2_idx2]],
                ],
                dim=1,
            )

            mixed = (lamda.unsqueeze(-1) * mixed).sum(dim=1, dtype=outputs.dtype)
            mixed_outputs[i, y1_label_ids[y1_idx]] = mixed

            y2_label_ids_flat = y2_label_ids[y2_idx][y2_idx2].flatten()
            lamda_flat = lamda[:, 1:].flatten()

            y2_idx3 = []
            for y2_label_id in y2_label_ids_flat.unique():
                mask = y2_label_ids_flat == y2_label_id
                idx = torch.randperm(mask.sum())[0]
                y2_idx3.append(torch.where(mask)[0][idx])
            y2_idx3 = torch.stack(y2_idx3)

            row, _ = np.unravel_index(
                y2_idx3.cpu(), y2_label_ids[y2_idx][y2_idx2].size()
            )

            mixed_outputs[i, y2_label_ids_flat[y2_idx3]] = mixed[row].clone()

            label_ids = torch.cat([y1_label_ids[y1_idx], y2_label_ids_flat[y2_idx3]])

            if not no_label_smoothing:
                mixed_batch_y[i, label_ids] = torch.cat(
                    [lamda[:, 0], lamda_flat[y2_idx3]]
                ).float()

        if flow_mixup_enabled:
            outputs = model((outputs, *others), **output_opts)[0]
            loss = criterion(outputs, batch_y)
        else:
            loss = 0
            flow_alpha = 1.0

        mixed_outputs = model((mixed_outputs, *others), **output_opts)[0]
        loss = loss + flow_alpha * criterion(mixed_outputs, mixed_batch_y)
        loss = loss / accumulation_step

    loss = scaler.scale(loss)
    loss.backward()

    if (global_step + 1) % accumulation_step == 0:
        scaler.unscale_(optim)
        clip_gradient(model, gradient_norm_queue, gradient_clip_value)

        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

    return loss.item()


def train_word_or_hidden_mixup_step(
    global_step: int,
    accumulation_step: int,
    model: nn.Module,
    criterion: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_y: torch.Tensor,
    scaler: GradScaler,
    optim: Optimizer,
    mixup_type: str,
    mixup_alpha: float,
    flow_mixup_enabled: bool,
    flow_alpha: float,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
):
    model.train()

    batch_size = batch_y.size(0)

    batch_x = to_device(batch_x, device)
    batch_y = to_device(batch_y, device)

    losses = []

    if mixup_type == "word":
        return_args = {"return_emb": True}
        pass_args = {"pass_emb": True}
    else:
        return_args = {"return_hidden": True}
        pass_args = {"pass_hidden": True}

    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        outputs = model(batch_x, **return_args)
        outputs, others = outputs[0], outputs[1:]

        mixed_outputs = outputs.clone()
        mixed_batch_y = batch_y.clone()

        ridx = torch.randperm(batch_size)
        lamda = np.random.beta(mixup_alpha, mixup_alpha)
        lamda = max(lamda, 1 - lamda)

        mixed_outputs = mixup(mixed_outputs, mixed_outputs[ridx], lamda)
        mixed_batch_y = mixup(mixed_batch_y, mixed_batch_y[ridx], lamda)

        if flow_mixup_enabled:
            outputs = model((outputs, *others), **pass_args)[0]
            losses.append(flow_alpha * criterion(outputs, batch_y))

        mixed_outputs = model((mixed_outputs, *others), **pass_args)[0]
        losses.append(criterion(mixed_outputs, mixed_batch_y))
        loss = torch.stack(losses).mean()
        loss = loss / accumulation_step

    loss = scaler.scale(loss)
    loss.backward()

    if (global_step + 1) % accumulation_step == 0:
        scaler.unscale_(optim)
        clip_gradient(model, gradient_norm_queue, gradient_clip_value)

        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

    return loss.item()


def train_in_place_mixup_step(
    global_step: int,
    accumulation_step: int,
    model: nn.Module,
    criterion: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_y: torch.Tensor,
    scaler: GradScaler,
    optim: Optimizer,
    mixup_num: int,
    mixup_alpha: float,
    flow_mixup_enabled: bool,
    flow_alpha: float,
    no_label_smoothing: bool,
    inv_w: np.ndarray,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
):
    model.train()

    batch_x = to_device(batch_x, device)
    batch_y = to_device(batch_y, device)

    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        outputs = model(batch_x, return_attn=True)
        outputs, others = outputs[0], outputs[1:]

        for i in range(batch_x.size(0)):
            label_ids = batch_y[i].nonzero(as_tuple=True)[0]

            if label_ids.size(0) > 1:
                mixed_outputs = outputs.clone()
                mixed_batch_y = batch_y.clone()

                while True:
                    target_idx = (
                        torch.tensor(1 - 1 / inv_w[label_ids.cpu()]).bernoulli().bool()
                    )

                    if len(target_idx.size()) == 0:
                        target_idx = target_idx.unsqueeze(0)

                    if target_idx.sum() > 0:
                        break

                mask = label_ids[target_idx].unsqueeze(-1) != label_ids.repeat(
                    (target_idx.sum(), 1)
                )
                scores = (
                    (outputs[i, label_ids[target_idx]] @ outputs[i, label_ids].T)
                    / np.sqrt(label_ids.size(-1))
                    * mask.to(device)
                )

                candidate_indices = scores.argsort(dim=-1, descending=True)

                n = min(mixup_num, label_ids.size(0))

                lamda = (
                    torch.distributions.Dirichlet(torch.tensor([mixup_alpha] * n))
                    .sample((target_idx.sum().item(),))
                    .to(device)
                    .to(outputs.dtype)
                    .sort(descending=True)[0]
                )

                mixed = torch.cat(
                    [
                        mixed_outputs[i, label_ids[target_idx]].unsqueeze(1),
                        mixed_outputs[i, label_ids[candidate_indices[:, : n - 1]]],
                    ],
                    dim=1,
                )
                mixed = (lamda.unsqueeze(-1) * mixed).sum(dim=1, dtype=outputs.dtype)
                mixed_outputs[i, label_ids[target_idx]] = mixed

                if not no_label_smoothing:
                    mixed_batch_y[i, label_ids[target_idx]] = lamda[:, 0].float()
            else:
                mixed_outputs = None
                mixed_batch_y = None

        if flow_mixup_enabled or mixed_outputs is None:
            outputs = model((outputs, *others), pass_attn=True)[0]
            loss = criterion(outputs, batch_y)
        else:
            loss = 0
            flow_alpha = 1.0

        if mixed_outputs is not None:
            mixed_outputs = model((mixed_outputs, *others), pass_attn=True)[0]
            loss = loss + flow_alpha * criterion(mixed_outputs, mixed_batch_y)

        loss = loss / accumulation_step

    loss = scaler.scale(loss)
    loss.backward()

    if (global_step + 1) % accumulation_step == 0:
        scaler.unscale_(optim)
        clip_gradient(model, gradient_norm_queue, gradient_clip_value)

        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

    return loss.item()


def train_in_place_mixup_step_v2(
    global_step: int,
    accumulation_step: int,
    model: nn.Module,
    criterion: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_y: torch.Tensor,
    scaler: GradScaler,
    optim: Optimizer,
    mixup_num: int,
    mixup_alpha: float,
    target_num: int,
    adj: csr_matrix,
    flow_mixup_enabled: bool,
    flow_alpha: float,
    no_label_smoothing: bool,
    inv_w: np.ndarray,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
):
    model.train()

    batch_size = batch_y.size(0)

    batch_x = to_device(batch_x, device)
    batch_y = to_device(batch_y, device)

    losses = []

    with_start = time.time()
    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        outputs = model(batch_x, return_attn=True)
        outputs, others = outputs[0], outputs[1:]

        emb_size = outputs.size(-1)

        mixed_outputs = outputs.clone()
        mixed_batch_y = batch_y.clone()

        outer_for_start = time.time()

        labels = torch.zeros(batch_size, target_num, mixup_num, dtype=torch.int64)
        masks = torch.zeros_like(labels, dtype=torch.bool)
        lamda = torch.ones_like(labels, dtype=torch.float)

        for i in range(batch_size):
            label_ids = batch_y[i].nonzero(as_tuple=True)[0]

            n = min(target_num, label_ids.size(0))
            selected_label_ids = np.random.choice(label_ids.cpu(), n, replace=False)

            inner_for_start = time.time()
            for j, label_id in enumerate(selected_label_ids):
                neigh = adj[label_id].indices

                if neigh.shape[0] == 0:
                    continue

                n = min(mixup_num - 1, len(neigh))

                selected_neigh = np.random.choice(
                    neigh,
                    size=n,
                    p=torch.tensor(inv_w[neigh]).softmax(dim=-1),
                    replace=False,
                )

                labels[i, j, 0] = label_id
                labels[i, j, 1 : n + 1] = torch.from_numpy(selected_neigh)
                masks[i, j, : n + 1] = True

                lam = (
                    torch.distributions.Dirichlet(torch.tensor([mixup_alpha] * (n + 1)))
                    .sample()
                    .sort(descending=True)[0]
                )

                lamda[i, j, : n + 1] = lam

                # mixed = torch.cat(
                #     [
                #         mixed_outputs[i, label_id].unsqueeze(0),
                #         mixed_outputs[i, selected_neigh],
                #     ],
                #     dim=0,
                # )
                # mixed = lamda.unsqueeze(1) * mixed  # N x D

                mixup_label_ids = [label_id] + selected_neigh.tolist()
                # mixed_outputs[i, mixup_label_ids] = mixed

                if not no_label_smoothing:
                    mixed_batch_y[i, mixup_label_ids] = lam.to(device)
                else:
                    mixed_batch_y[i, mixup_label_ids] = 1.0

            inner_for_end = time.time()

        indices = labels.view(batch_size, -1, 1).repeat(1, 1, emb_size).to(device)
        mixed = torch.gather(mixed_outputs, 1, indices, sparse_grad=True).view(
            batch_size, target_num, mixup_num, -1
        )
        mixed = lamda.unsqueeze(-1).to(device).to(mixed.dtype) * mixed

        mixed_outputs = mixed_outputs.scatter(
            1, indices, mixed.view(batch_size, -1, emb_size)
        )

        outer_for_end = time.time()

        # print(f"outer_for: {(outer_for_end - outer_for_start) * 1000:.2f} ms")
        # print(f"inner_for: {(inner_for_end - inner_for_start) * 1000:.2f} ms")

        if flow_mixup_enabled:
            outputs = model((outputs, *others), pass_attn=True)[0]
            losses.append(flow_alpha * criterion(outputs, batch_y))

        forward_start = time.time()
        mixed_outputs = model((mixed_outputs, *others), pass_attn=True)[0]
        forward_end = time.time()

        # print(f"forward: {(forward_end - forward_start) * 1000:.2f} ms")

        loss_start = time.time()
        losses.append(criterion(mixed_outputs, mixed_batch_y))
        loss = torch.stack(losses).mean()
        loss = loss / accumulation_step
        loss_end = time.time()

        # print(f"loss: {(loss_end - loss_start) * 1000:.2f} ms")

    with_end = time.time()

    # print(f"with: {(with_end - with_start) * 1000:.2f} ms")

    optim_start = time.time()

    loss = scaler.scale(loss)

    backward_start = time.time()
    loss.backward()
    backward_end = time.time()

    # print(f"backward: {(backward_end - backward_start) * 1000:.2f} ms")

    if (global_step + 1) % accumulation_step == 0:
        scaler.unscale_(optim)
        clip_gradient(model, gradient_norm_queue, gradient_clip_value)

        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

    optim_end = time.time()

    # print(list(model.parameters())[2])
    # print(f"optim: {(optim_end - optim_start) * 1000:.2f} ms")

    return loss.item()


def train_ssmix_step(
    model: nn.Module,
    criterion: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_y: torch.Tensor,
    scaler: GradScaler,
    optim: Optimizer,
    ss_winsize: float,
    naive_augment: bool = False,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
) -> Optional[float]:
    model.train()

    batch_x = to_device(batch_x, device)
    batch_y = to_device(batch_y, device)

    inputs_left, targets_left, inputs_right, targets_right = split_batch(
        batch_x, batch_y
    )

    # Skip odd-numbered batch
    if inputs_left is None:
        return 0

    losses = []
    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        if not naive_augment:
            losses.append(get_loss(model, criterion, inputs_left, targets_left))
            losses.append(get_loss(model, criterion, inputs_right, targets_right))

        inputs_aug_left, ratio_left = ssmix_augment(
            model,
            criterion,
            inputs_left,
            inputs_right,
            targets_left,
            targets_right,
            ss_winsize,
        )
        inputs_aug_right, ratio_right = ssmix_augment(
            model,
            criterion,
            inputs_right,
            inputs_left,
            targets_right,
            targets_left,
            ss_winsize,
        )

        losses.append(
            get_loss(
                model,
                criterion,
                inputs_aug_left,
                targets_left,
                targets_right,
                ratio_left,
            )
        )
        losses.append(
            get_loss(
                model,
                criterion,
                inputs_aug_right,
                targets_right,
                targets_left,
                ratio_right,
            )
        )

    loss = scaler.scale(torch.stack(losses).mean())
    loss.backward()
    scaler.unscale_(optim)
    clip_gradient(model, gradient_norm_queue, gradient_clip_value)
    scaler.step(optim)
    optim.zero_grad()

    return loss.item()


# Reference: https://github.com/clovaai/ssmix
def train_embed_or_tmix_step(
    global_step: int,
    accumulation_step: int,
    model: nn.Module,
    criterion: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_y: torch.Tensor,
    scaler: GradScaler,
    optim: Optimizer,
    mixup_type: str,
    mixup_alpha: float,
    naive_augment: bool = False,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
) -> Optional[float]:
    model.train()

    batch_x = to_device(batch_x, device)
    batch_y = to_device(batch_y, device)

    if mixup_type == "tmix":
        mix_layer = random.choice([7, 9, 12]) - 1
        mix_embedding = False
    elif mixup_type == "embedmix":
        mix_embedding = True
        mix_layer = None
    else:
        raise ValueError(f"Invalid mixup type: {mixup_type}")

    l = np.random.beta(mixup_alpha, mixup_alpha)
    l = max(l, 1 - l)

    inputs1, targets1, inputs2, targets2 = split_batch(batch_x, batch_y)

    if inputs1 is None:
        return 0

    losses = []
    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        if not naive_augment:
            losses.append(get_loss(model, criterion, inputs1, targets1))
            losses.append(get_loss(model, criterion, inputs2, targets2))

        losses.append(
            get_loss(
                model,
                criterion,
                inputs1,
                targets1,
                targets2,
                l,
                inputs2=inputs2,
                mix_lambda=l,
                mix_layer=mix_layer,
                mix_embedding=mix_embedding,
            )
        )
        losses.append(
            get_loss(
                model,
                criterion,
                inputs2,
                targets2,
                targets1,
                l,
                inputs2=inputs1,
                mix_lambda=l,
                mix_layer=mix_layer,
                mix_embedding=mix_embedding,
            )
        )

    loss = scaler.scale(torch.stack(losses).mean())
    loss.backward()

    if (global_step + 1) % accumulation_step == 0:
        scaler.unscale_(optim)
        clip_gradient(model, gradient_norm_queue, gradient_clip_value)

        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

    return loss.item()


def predict_step(
    model: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    topk: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    with torch.no_grad():
        logits = model(to_device(batch_x, device), mp_enabled=False)[0]

    scores, labels = torch.topk(logits, topk)
    scores = torch.sigmoid(scores)

    return scores.cpu(), labels.cpu()


def get_results(
    model: nn.Module,
    dataloader: DataLoader,
    raw_y: np.ndarray,
    mlb: MultiLabelBinarizer,
    inv_w: Optional[np.ndarray] = None,
    is_test: bool = False,
    train_labels: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(raw_y)

    model.eval()

    sum_ranking = np.zeros(len(mlb.classes_), dtype=np.int)
    scores = []
    labels = []
    coverage = []
    for batch_x, batch_y in tqdm(dataloader, desc="Predict", leave=False):
        with torch.no_grad():
            logits = model(to_device(batch_x, device), mp_enabled=False)[0]
            ranking = logits.argsort(descending=True, dim=-1).cpu()

            y_scores = logits.cpu()
            y_scores_mask = y_scores.masked_fill(~batch_y.bool(), np.inf)
            y_min_relevant = y_scores_mask.min(dim=-1)[0].view(-1, 1)
            coverage.append((y_scores >= y_min_relevant).sum(dim=-1))

            s, l = torch.topk(logits, 50)
            s = torch.sigmoid(s)

            scores.append(s.cpu())
            labels.append(l.cpu())

            for i in range(batch_y.size(0)):
                label_idx = batch_y[i].nonzero(as_tuple=True)[0]
                rank = torch.cat(
                    [(ranking[i] == l).nonzero(as_tuple=True)[0] for l in label_idx]
                )
                sum_ranking[label_idx] += rank.numpy() + 1

    with np.errstate(divide="ignore", invalid="ignore"):
        if isinstance(dataloader.dataset, Subset):
            y = dataloader.dataset.dataset.y[dataloader.dataset.indices]
        else:
            y = dataloader.dataset.y

        avg_ranking = sum_ranking / y.sum(axis=0).A1
        np.nan_to_num(avg_ranking, copy=False)

    coverage = np.concatenate(coverage)
    prediction = mlb.classes_[np.concatenate(labels)]

    if is_test:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(raw_y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inv_w = get_inv_propensity(mlb.transform(train_labels))
        results = get_precision_results2(prediction, raw_y, inv_w, mlb)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            p5 = get_p_5(prediction, raw_y, mlb)
            n5 = get_n_5(prediction, raw_y, mlb)
            r10 = get_r_10(prediction, raw_y, mlb)

            results = {
                "p5": p5,
                "n5": n5,
                "r10": r10,
            }

            if inv_w is not None:
                psp5 = get_psp_5(prediction, raw_y, inv_w, mlb)
                results["psp5"] = psp5

    results["avg.rank"] = avg_ranking.mean()
    results["coverage"] = coverage.mean()

    return results


def get_optimizer(
    model: nn.Module,
    lr: float,
    decay: float,
    classifier_lr: float = None,
    classifier_decay: float = None,
) -> Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    if classifier_lr is None:
        classifier_lr = lr

    if classifier_decay is None:
        classifier_decay = decay

    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not (
                    re.match(r"^(module\.)?attention", n)
                    or re.match(r"^(module\.)?linear", n)
                )
            ],
            "weight_decay": decay,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and not (
                    re.match(r"^(module\.)?attention", n)
                    or re.match(r"^(module\.)?linear", n)
                )
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and (
                    re.match(r"^(module\.)?attention", n)
                    or re.match(r"^(module\.)?linear", n)
                )
            ],
            "weight_decay": classifier_decay,
            "lr": classifier_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
                and (
                    re.match(r"^(module\.)?attention", n)
                    or re.match(r"^(module\.)?linear", n)
                )
            ],
            "weight_decay": 0.0,
            "lr": classifier_lr,
        },
    ]

    return DenseSparseAdamW(param_groups)


@click.command(context_settings={"show_default": True})
@click.option(
    "--mode",
    type=click.Choice(["train", "eval"]),
    default="train",
    help="train: train and eval are executed. eval: eval only",
)
@click.option(
    "--eval-ckpt-path",
    type=click.Path(exists=True),
    help="ckpt path to be evaludated. Only available eval mode",
)
@click.option("--test-run", is_flag=True, default=False, help="Test run mode for debug")
@click.option(
    "--run-script", type=click.Path(exists=True), help="Run script file path to log"
)
@click.option("--seed", type=click.INT, default=0, help="Seed for reproducibility")
@click.option(
    "--model-cnf", type=click.Path(exists=True), help="Model config file path"
)
@click.option("--data-cnf", type=click.Path(exists=True), help="Data config file path")
@click.option(
    "--ckpt-root-path",
    type=click.Path(),
    default="./checkpoint",
    help="Checkpoint root path",
)
@click.option("--ckpt-name", type=click.STRING, help="Checkpoint name")
@click.option(
    "--ckpt-epoch",
    type=click.INT,
    multiple=True,
    help="Specify epoch for saving checkpoints",
)
@click.option(
    "--ckpt-save-interval",
    type=click.INT,
    help="Set checkpoint saving interval based on epoch",
)
@click.option(
    "--mp-enabled", is_flag=True, default=False, help="Enable Mixed Precision"
)
@click.option(
    "--swa-warmup", type=click.INT, default=10, help="Warmup for SWA. Disable: -1"
)
@click.option(
    "--eval-step",
    type=click.INT,
    default=100,
    help="Evaluation step during training",
)
@click.option("--print-step", type=click.INT, default=20, help="Print step")
@click.option(
    "--early",
    type=click.INT,
    default=50,
    help="Early stopping step",
)
@click.option(
    "--early-criterion",
    type=click.Choice(["p5", "n5", "psp5"]),
    default="n5",
    help="Early stopping criterion",
)
@click.option("--reset-best", is_flag=True, default=False, help="Reset best")
@click.option(
    "--num-epochs", type=click.INT, default=200, help="Total number of epochs"
)
@click.option(
    "--train-batch-size", type=click.INT, default=128, help="Batch size for training"
)
@click.option(
    "--test-batch-size", type=click.INT, default=256, help="Batch size for test"
)
@click.option("--no-cuda", is_flag=True, default=False, help="Disable cuda")
@click.option(
    "--num-workers", type=click.INT, default=4, help="Number of workers for data loader"
)
@click.option(
    "--decay",
    type=click.FLOAT,
    default=1e-2,
    help="Weight decay (Base Encoder)",
)
@click.option(
    "--lr", type=click.FLOAT, default=1e-3, help="learning rate (Base Encoder)"
)
@click.option(
    "--classifier-lr",
    type=click.FLOAT,
    help="learning rate for classifier",
)
@click.option(
    "--classifier-decay",
    type=click.FLOAT,
    help="weight decay for classifier",
)
@click.option(
    "--accumulation-step",
    type=click.INT,
    default=1,
    help="accumlation step for small batch size",
)
@click.option(
    "--enable-loss-weight",
    is_flag=True,
    default=False,
    help="Enable loss weights for BCE loss",
)
@click.option("--resume", is_flag=True, default=False, help="Resume training")
@click.option("--resume-ckpt-path", type=click.STRING, help="ckpt for resume training")
@click.option(
    "--gradient-max-norm",
    type=click.FLOAT,
    help="max norm for gradient clipping",
)
@optgroup.group("Common Mixup options")
@optgroup.option(
    "--mixup-enabled",
    is_flag=True,
    default=False,
    help="Enable mixup",
)
@optgroup.option(
    "--mixup-lr",
    type=click.FLOAT,
    help="Learning rate for mixup",
)
@optgroup.option(
    "--mixup-decay",
    type=click.FLOAT,
    help="Weight decay for mixup",
)
@optgroup.option(
    "--mixup-type",
    type=click.Choice(
        ["word", "hidden", "inplace", "inplace2", "ssmix", "embedmix", "tmix"]
    ),  # word == embedmix, hidden == tmix
    default="inplace2",
    help="Type of Mixup",
)
@optgroup.option(
    "--mixup-alpha", type=click.FLOAT, default=0.4, help="Hyper parameter for mixup"
)
@optgroup.option(
    "--mixup-warmup",
    type=click.INT,
    default=20,
    help="Deferred stragtegy for mixup. Disable: -1",
)
@optgroup.option(
    "--no-label-smoothing", is_flag=True, default=False, help="No label smoothing"
)
@optgroup.option(
    "--mixup-num", type=click.INT, default=2, help="# of samples to be mixed"
)
@optgroup.group("in-place mixup options")
@optgroup.option(
    "--in-place-target-num",
    type=click.INT,
    default=3,
    help="# of target for in-place mixup v2",
)
@optgroup.group("ssmix options")
@optgroup.option(
    "--ss-winsize",
    type=click.FloatRange(0, 1),
    default=0.1,
    help="Percent of window size for augmentation",
)
@optgroup.option(
    "--ss-no-saliency",
    is_flag=True,
    default=False,
    help="Excluding saliency constraint in SSMix",
)
@optgroup.option(
    "--ss-no-span",
    is_flag=True,
    default=False,
    help="Excluding span constraint in SSMix",
)
@optgroup.option(
    "--naive-augment", is_flag=True, default=False, help="Augment without original data"
)
@optgroup.group("flow mixup options")
@optgroup.option(
    "--flow-mixup-enabled", is_flag=True, default=False, help="Enable flow mixup"
)
@optgroup.option(
    "--flow-alpha", type=click.FLOAT, default=1.0, help="ratio of loss of mixed data"
)
@log_elapsed_time
def main(
    mode: str,
    eval_ckpt_path: str,
    test_run: bool,
    run_script: str,
    seed: int,
    model_cnf: str,
    data_cnf: str,
    ckpt_root_path: str,
    ckpt_name: str,
    ckpt_epoch: List[int],
    ckpt_save_interval: int,
    mp_enabled: bool,
    swa_warmup: int,
    eval_step: int,
    print_step: int,
    early: int,
    early_criterion: str,
    reset_best: bool,
    num_epochs: int,
    train_batch_size: int,
    test_batch_size: int,
    no_cuda: bool,
    num_workers: int,
    decay: float,
    lr: float,
    classifier_lr: float,
    classifier_decay: float,
    accumulation_step: int,
    enable_loss_weight: bool,
    resume: bool,
    resume_ckpt_path: str,
    gradient_max_norm: float,
    mixup_enabled: bool,
    mixup_lr: float,
    mixup_decay: float,
    mixup_type: str,  # word == embedmix
    mixup_alpha: float,
    mixup_warmup: int,
    no_label_smoothing: bool,
    mixup_num: int,
    in_place_target_num: int,
    ss_winsize: float,
    ss_no_saliency: bool,
    ss_no_span: bool,
    naive_augment: bool,
    flow_mixup_enabled: bool,
    flow_alpha: float,
):
    yaml = YAML(typ="safe")

    model_cnf_path = model_cnf
    data_cnf_path = data_cnf

    model_cnf = yaml.load(Path(model_cnf))
    data_cnf = yaml.load(Path(data_cnf))

    model_name = model_cnf["name"]
    dataset_name = data_cnf["name"]

    ################################ Assert options ##################################
    if mixup_type in ["ssmix", "embedmix", "tmix"]:
        assert model_name in MIX_MODELS
        if model_name == "AttentionRNN4Mix":
            assert mixup_type == "ssmix", f"{model_name} does not support {mixup_type}"

    if mixup_type == "ssmix":
        assert accumulation_step == 1, "ssmix does not support gradient accumulation"
        assert not mp_enabled, "ssmix does not support mixed precision"

    if mixup_type in ["word", "hidden"]:
        if no_label_smoothing:
            logger.warning(f"{mixup_type} is not support no_label_smoothing option.")
    ##################################################################################

    ################################ Initialize Config ###############################
    prefix = "" if ckpt_name is None else f"{ckpt_name}_"
    ckpt_name = f"{prefix}{model_name}_{dataset_name}_{seed}"
    ckpt_root_path = os.path.join(ckpt_root_path, ckpt_name)
    ckpt_path = os.path.join(ckpt_root_path, "ckpt.pt")
    last_ckpt_path = os.path.join(ckpt_root_path, "ckpt.last.pt")
    log_filename = "train.log"

    os.makedirs(ckpt_root_path, exist_ok=True)

    if (
        (not resume or resume_ckpt_path)
        and os.path.exists(ckpt_path)
        and mode == "train"
    ):
        click.confirm(
            f"Checkpoint is already existed. ({ckpt_path})\nOverwrite it?",
            abort=True,
            err=True,
        )

        if not resume_ckpt_path:
            shutil.rmtree(ckpt_root_path)
            os.makedirs(ckpt_root_path, exist_ok=True)

    if not test_run and mode == "train":
        set_logger(os.path.join(ckpt_root_path, log_filename))

        copy_file(
            model_cnf_path,
            os.path.join(ckpt_root_path, os.path.basename(model_cnf_path)),
        )
        copy_file(
            data_cnf_path, os.path.join(ckpt_root_path, os.path.basename(data_cnf_path))
        )

        if run_script is not None:
            copy_file(
                run_script, os.path.join(ckpt_root_path, os.path.basename(run_script))
            )

    if seed is not None:
        logger.info(f"seed: {seed}")
        set_seed(seed)

    device = torch.device("cpu" if no_cuda else "cuda")
    num_gpus = torch.cuda.device_count()
    ##################################################################################

    ################################ Prepare Dataset #################################
    logger.info(f"Dataset: {dataset_name}")

    train_dataset, test_dataset, train_ids, valid_ids = get_dataset(model_cnf, data_cnf)
    inv_w = get_inv_propensity(train_dataset.y)
    mlb = get_mlb(train_dataset.le_path)
    num_labels = len(mlb.classes_)

    if mixup_enabled and mixup_type == "inplace2":
        adj = lil_matrix(train_dataset.y.T @ train_dataset.y)
        adj.setdiag(0)
        adj = adj.tocsr()
        adj.eliminate_zeros()
    else:
        adj = None

    logger.info(f"# of train dataset: {train_ids.shape[0]:,}")
    logger.info(f"# of valid dataset: {valid_ids.shape[0]:,}")
    logger.info(f"# of test dataset: {len(test_dataset):,}")
    logger.info(f"# of labels: {num_labels:,}")
    ##################################################################################

    ################################# Prepare Model ##################################
    logger.info(f"Model: {model_name}")

    model = get_model(model_cnf, data_cnf, num_labels, mp_enabled, device)

    if num_gpus > 1 and not no_cuda:
        logger.info(f"Multi-GPU mode: {num_gpus} GPUs")
        model = nn.DataParallel(model)
    elif not no_cuda:
        logger.info("Single-GPU mode")
    else:
        logger.info("CPU mode")
    ##################################################################################

    ############################### Prepare Training #################################
    optimizer = get_optimizer(model, lr, decay, classifier_lr, classifier_decay)

    scheduler = None
    scaler = GradScaler(enabled=mp_enabled)

    loss_weight = torch.tensor(inv_w) if enable_loss_weight else None
    criterion = nn.BCEWithLogitsLoss(weight=loss_weight)
    criterion.to(device)

    gradient_norm_queue = (
        deque([np.inf], maxlen=5) if gradient_max_norm is not None else None
    )

    model_swa_state = {}
    results = {}

    start_epoch = 0
    global_step = 0
    best, e = 0, 0
    early_stop = False

    train_losses = deque(maxlen=print_step)

    if resume and mode == "train":
        if resume_ckpt_path is None:
            resume_ckpt_path = (
                last_ckpt_path if os.path.exists(last_ckpt_path) else ckpt_path
            )

        if os.path.exists(resume_ckpt_path):
            logger.info("Resume Training")
            start_epoch, ckpt = load_checkpoint2(
                resume_ckpt_path,
                [model],
                # optim=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                set_rng_state=True,
                return_other_states=True,
            )

            start_epoch += 1
            epoch = start_epoch
            global_step = ckpt["global_step"]
            gradient_norm_queue = ckpt["gradient_norm_queue"]
            model_swa_state = ckpt["model_swa_state"]
            best = ckpt["best"] if not reset_best else 0
            e = ckpt["e"]

        else:
            logger.warning("No checkpoint")
    ##################################################################################

    ############################### Prepare Dataloader ###############################
    logger.info(f"Prepare Dataloader")

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(
        model_cnf,
        train_dataset,
        test_dataset,
        train_ids,
        valid_ids,
        train_batch_size,
        test_batch_size,
        num_workers,
        no_cuda,
    )
    ##################################################################################

    logger.info(f"checkpoint name: {os.path.basename(ckpt_name)}")

    ##################################### Training ###################################
    input_opts = model_cnf["train"]["input_opts"]
    output_opts = model_cnf["train"]["output_opts"]
    results = None

    if mode == "train":
        try:
            for epoch in range(start_epoch, num_epochs):
                if early_stop:
                    break

                if epoch == swa_warmup:
                    logger.info("Initialze SWA")
                    swa_init(model, model_swa_state)

                if epoch == mixup_warmup and mixup_enabled:
                    logger.info("Start Mixup")
                    mixup_ckpt_path, ext = os.path.splitext(ckpt_path)
                    mixup_ckpt_path += "_before_mixup" + ext

                    if reset_best:
                        best = 0

                    save_checkpoint2(
                        mixup_ckpt_path,
                        epoch,
                        [model],
                        optim=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        results=results,
                        other_states={
                            "best": best,
                            "train_ids": train_ids,
                            "valid_ids": valid_ids,
                            "model_swa_state": model_swa_state,
                            "global_step": global_step,
                            "early_criterion": early_criterion,
                            "gradient_norm_queue": gradient_norm_queue,
                            "e": e,
                        },
                    )

                    if mixup_lr is not None:
                        set_lr(optimizer, mixup_lr, mixup_decay)

                if resume and mixup_enabled and epoch == mixup_warmup + 1:
                    logger.info("Start Mixup")

                    if mixup_lr is not None:
                        set_lr(optimizer, mixup_lr, mixup_decay)

                for i, (batch_x, batch_y) in enumerate(train_dataloader, 1):
                    if epoch >= mixup_warmup and mixup_enabled:
                        if mixup_type in ["word", "hidden"]:
                            train_loss = train_word_or_hidden_mixup_step(
                                global_step,
                                accumulation_step,
                                model,
                                criterion,
                                batch_x,
                                batch_y,
                                scaler,
                                optimizer,
                                mixup_type,
                                mixup_alpha,
                                flow_mixup_enabled,
                                flow_alpha,
                                gradient_max_norm,
                                gradient_norm_queue,
                                device,
                            )
                        elif mixup_type == "inplace":
                            train_loss = train_in_place_mixup_step(
                                global_step,
                                accumulation_step,
                                model,
                                criterion,
                                batch_x,
                                batch_y,
                                scaler,
                                optimizer,
                                mixup_num,
                                mixup_alpha,
                                flow_mixup_enabled,
                                flow_alpha,
                                no_label_smoothing,
                                inv_w,
                                gradient_max_norm,
                                gradient_norm_queue,
                                device,
                            )
                        elif mixup_type == "inplace2":
                            step_start = time.time()
                            train_loss = train_in_place_mixup_step_v2(
                                global_step,
                                accumulation_step,
                                model,
                                criterion,
                                batch_x,
                                batch_y,
                                scaler,
                                optimizer,
                                mixup_num,
                                mixup_alpha,
                                in_place_target_num,
                                adj,
                                flow_mixup_enabled,
                                flow_alpha,
                                no_label_smoothing,
                                inv_w,
                                gradient_max_norm,
                                gradient_norm_queue,
                                device,
                            )
                            step_end = time.time()
                            # print(f"step: {(step_end - step_start) * 1000:.2f} ms")
                        elif mixup_type == "ssmix":
                            train_loss = train_ssmix_step(
                                model,
                                criterion,
                                batch_x,
                                batch_y,
                                scaler,
                                optimizer,
                                ss_winsize,
                                naive_augment,
                                gradient_max_norm,
                                gradient_norm_queue,
                                device,
                            )
                        elif mixup_type in ["embedmix", "tmix"]:
                            train_loss = train_embed_or_tmix_step(
                                global_step,
                                accumulation_step,
                                model,
                                criterion,
                                batch_x,
                                batch_y,
                                scaler,
                                optimizer,
                                mixup_type,
                                mixup_alpha,
                                naive_augment,
                                gradient_max_norm,
                                gradient_norm_queue,
                                device,
                            )
                        else:
                            train_loss = train_mixup_step(
                                global_step,
                                accumulation_step,
                                model,
                                criterion,
                                batch_x,
                                batch_y,
                                scaler,
                                optimizer,
                                mixup_num,
                                mixup_alpha,
                                flow_mixup_enabled,
                                no_label_smoothing,
                                input_opts,
                                output_opts,
                                inv_w,
                                gradient_max_norm,
                                gradient_norm_queue,
                                device,
                            )
                    else:
                        step_start = time.time()
                        train_loss = train_step(
                            global_step,
                            accumulation_step,
                            model,
                            criterion,
                            batch_x,
                            batch_y,
                            scaler,
                            optimizer,
                            gradient_clip_value=gradient_max_norm,
                            gradient_norm_queue=gradient_norm_queue,
                            device=device,
                        )
                        step_end = time.time()
                        # print(f"step: {(step_end - step_start) * 1000:.2f} ms")

                    if scheduler is not None:
                        scheduler.step()

                    train_losses.append(train_loss)

                    global_step += 1

                    val_log_msg = ""
                    if global_step % eval_step == 0 or (
                        epoch == num_epochs - 1 and i == len(train_dataloader)
                    ):
                        swa_step(model, model_swa_state)
                        swap_swa_params(model, model_swa_state)

                        results = get_results(
                            model,
                            valid_dataloader,
                            train_dataset.raw_y[valid_ids],
                            mlb=mlb,
                            inv_w=inv_w,
                            device=device,
                        )

                        val_log_msg = (
                            f"p@5: {results['p5']:.5f} n@5: {results['n5']:.5f} "
                        )

                        if "psp5" in results:
                            val_log_msg += f"psp@5: {results['psp5']:.5f} "

                        val_log_msg += f"avg.rank: {results['avg.rank']:.2f} "
                        val_log_msg += f"coverage: {results['coverage']:.2f}"

                        if best < results[early_criterion]:
                            best = results[early_criterion]
                            e = 0
                            save_checkpoint2(
                                ckpt_path,
                                epoch,
                                [model],
                                optim=optimizer,
                                scaler=scaler,
                                scheduler=scheduler,
                                results=results,
                                other_states={
                                    "best": best,
                                    "train_ids": train_ids,
                                    "valid_ids": valid_ids,
                                    "model_swa_state": model_swa_state,
                                    "global_step": global_step,
                                    "early_criterion": early_criterion,
                                    "gradient_norm_queue": gradient_norm_queue,
                                    "e": e,
                                },
                            )
                        else:
                            e += 1

                        swap_swa_params(model, model_swa_state)

                    if (
                        global_step % print_step == 0
                        or global_step % eval_step == 0
                        or (epoch == num_epochs - 1 and i == len(train_dataloader))
                    ):
                        log_msg = f"{epoch} {i * train_dataloader.batch_size} "
                        log_msg += f"early stop: {e} "
                        log_msg += f"train loss: {np.mean(train_losses):.5f} "
                        log_msg += val_log_msg

                        logger.info(log_msg)

                        if early is not None and e > early:
                            early_stop = True
                            break

                if ckpt_epoch is not None and epoch in ckpt_epoch:
                    ckpt_epoch_path = os.path.join(ckpt_root_path, f"ckpt.{epoch}.pt")
                    save_checkpoint2(
                        ckpt_epoch_path,
                        epoch,
                        [model],
                        optim=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        results=results,
                        other_states={
                            "best": best,
                            "train_ids": train_ids,
                            "valid_ids": valid_ids,
                            "model_swa_state": model_swa_state,
                            "global_step": global_step,
                            "early_criterion": early_criterion,
                            "gradient_norm_queue": gradient_norm_queue,
                            "e": e,
                        },
                    )

                if (
                    ckpt_save_interval is not None
                    and (epoch + 1) % ckpt_save_interval == 0
                ):
                    ckpt_epoch_path = os.path.join(ckpt_root_path, f"ckpt.{epoch}.pt")
                    if not os.path.exists(ckpt_epoch_path):
                        save_checkpoint2(
                            ckpt_epoch_path,
                            epoch,
                            [model],
                            optim=optimizer,
                            scaler=scaler,
                            scheduler=scheduler,
                            results=results,
                            other_states={
                                "best": best,
                                "train_ids": train_ids,
                                "valid_ids": valid_ids,
                                "model_swa_state": model_swa_state,
                                "global_step": global_step,
                                "early_criterion": early_criterion,
                                "gradient_norm_queue": gradient_norm_queue,
                                "e": e,
                            },
                        )

        except KeyboardInterrupt:
            logger.info("Interrupt training.")

        save_checkpoint2(
            last_ckpt_path,
            epoch,
            [model],
            optim=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            results=results,
            other_states={
                "best": best,
                "train_ids": train_ids,
                "valid_ids": valid_ids,
                "model_swa_state": model_swa_state,
                "global_step": global_step,
                "early_criterion": early_criterion,
                "gradient_norm_queue": gradient_norm_queue,
                "e": e,
            },
        )
    ##################################################################################

    ################################### Evaluation ###################################
    logger.info("Evaluation.")

    if mode == "eval" and eval_ckpt_path is not None:
        ckpt_path = eval_ckpt_path

    if os.path.exists(ckpt_path):
        load_checkpoint2(ckpt_path, [model], set_rng_state=False)

    results = get_results(
        model,
        test_dataloader,
        test_dataset.raw_y,
        mlb,
        inv_w,
        True,
        train_dataset.raw_y,
        device,
    )

    logger.info(
        f"\np@1,3,5,10: {results['p1']:.4f}, {results['p3']:.4f}, {results['p5']:.4f}, {results['p10']:.4f}"
        f"\nn@1,3,5,10: {results['n1']:.4f}, {results['n3']:.4f}, {results['n5']:.4f}, {results['n10']:.4f}"
        f"\npsp@1,3,5,10,20: {results['psp1']:.4f}, {results['psp3']:.4f}, {results['psp5']:.4f}, {results['psp10']:.4f}, {results['psp20']:.4f}"
        f"\npsn@1,3,5,10,20: {results['psn1']:.4f}, {results['psn3']:.4f}, {results['psn5']:.4f}, {results['psn10']:.4f}, {results['psn20']:.4f}"
        # f"\nr@1,5,10: {results['r1']:.4f}, {results['r5']:.4f}, {results['r10']:.4f}"
        f"\navg.rank: {results['avg.rank']:.2f} "
        f"\ncoverage: {results['coverage']:.2f}"
    )
    logger.info(f"checkpoint name: {os.path.basename(ckpt_name)}")
    ##################################################################################


if __name__ == "__main__":
    main()
