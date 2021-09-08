"""
Created on 2021/07/06
@author Sangwoo Han

Instace Anchor new version, no cluster
"""
import copy
import os
import shutil
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch.nn as nn
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
from transformers import AutoTokenizer

from m2m_text.datasets import (
    AmazonCat,
    AmazonCat13K,
    EURLex,
    EURLex4K,
    Wiki10,
    Wiki10_31K,
)
from m2m_text.datasets.sbert import SBertDataset, collate_fn
from m2m_text.datasets.text import TextDataset
from m2m_text.metrics import (
    get_inv_propensity,
    get_n_5,
    get_p_5,
    get_precision_results2,
    get_psp_5,
    get_r_10,
)
from m2m_text.networks import AttentionRNN, LaRoberta
from m2m_text.optimizers import DenseSparseAdamW
from m2m_text.utils.data import copy_file, get_mlb
from m2m_text.utils.mixup import MixUp, mixup
from m2m_text.utils.model import load_checkpoint2, save_checkpoint2
from m2m_text.utils.train import (
    clip_gradient,
    get_avg_ranking,
    log_elapsed_time,
    normalize_inv_w,
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
    "LaRoberta": LaRoberta,
}

TRANSFORMER_MODELS = ["LaRoberta"]


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
        model_name = model_cnf["model"].pop("model_name")
        model = (
            MODEL_CLS[model_name]
            .from_pretrained(
                model_name,
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
        train_texts = train_dataset.raw_data()[0]
        test_texts = test_dataset.raw_data()[0]

        tokenizer = AutoTokenizer.from_pretrained(model_cnf["model"]["model_name"])

        train_sbert_dataset = SBertDataset(
            tokenizer(
                [s.strip() for s in train_texts[train_ids]], **model_cnf["tokenizer"]
            ),
            train_dataset.y[train_ids],
        )
        valid_sbert_dataset = SBertDataset(
            tokenizer(
                [s.strip() for s in train_texts[valid_ids]], **model_cnf["tokenizer"]
            ),
            train_dataset.y[valid_ids],
        )
        test_sbert_dataset = SBertDataset(
            tokenizer([s.strip() for s in test_texts], **model_cnf["tokenizer"]),
            test_dataset.y,
        )

        train_dataloader = DataLoader(
            train_sbert_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        valid_dataloader = DataLoader(
            valid_sbert_dataset,
            batch_size=test_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        test_dataloader = DataLoader(
            test_sbert_dataset,
            batch_size=test_batch_size,
            collate_fn=collate_fn,
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


def train_step(
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

    optim.zero_grad()

    scaler.scale(loss).backward()
    scaler.unscale_(optim)
    clip_gradient(model, gradient_norm_queue, gradient_clip_value)
    scaler.step(optim)
    scaler.update()

    return loss.item()


def train_mixup_step(
    model: nn.Module,
    criterion: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_y: torch.Tensor,
    scaler: GradScaler,
    optim: Optimizer,
    mixup_num: int,
    mixup_alpha: float,
    flow_mixup_enabled: bool,
    no_label_smoothing: bool,
    input_opts: Dict[str, Any],
    output_opts: Dict[str, Any],
    inv_w: np.ndarray,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
):
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

        mixed_outputs = model((mixed_outputs, *others), **output_opts)[0]
        loss = loss + criterion(mixed_outputs, mixed_batch_y)

    optim.zero_grad()
    scaler.scale(loss).backward()
    clip_gradient(model, gradient_norm_queue, gradient_clip_value)
    scaler.step(optim)
    scaler.update()

    return loss.item()


def train_in_place_mixup_step(
    model: nn.Module,
    criterion: nn.Module,
    batch_x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch_y: torch.Tensor,
    scaler: GradScaler,
    optim: Optimizer,
    mixup_num: int,
    mixup_alpha: float,
    flow_mixup_enabled: bool,
    no_label_smoothing: bool,
    input_opts: Dict[str, Any],
    output_opts: Dict[str, Any],
    inv_w: np.ndarray,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
):
    model.train()

    batch_x = to_device(batch_x, device)
    batch_y = to_device(batch_y, device)

    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        outputs = model(batch_x, **input_opts)
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
            outputs = model((outputs, *others), **output_opts)[0]
            loss = criterion(outputs, batch_y)
        else:
            loss = 0

        if mixed_outputs is not None:
            mixed_outputs = model((mixed_outputs, *others), **output_opts)[0]
            loss = loss + criterion(mixed_outputs, mixed_batch_y)

    optim.zero_grad()
    scaler.scale(loss).backward()
    clip_gradient(model, gradient_norm_queue, gradient_clip_value)
    scaler.step(optim)
    scaler.update()

    return loss.item()


def train_in_place_mixup_step_v2(
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
    no_label_smoothing: bool,
    input_opts: Dict[str, Any],
    output_opts: Dict[str, Any],
    inv_w: np.ndarray,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
):
    model.train()

    batch_x = to_device(batch_x, device)
    batch_y = to_device(batch_y, device)

    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        outputs = model(batch_x, **input_opts)
        outputs, others = outputs[0], outputs[1:]

        mixed_outputs = outputs.clone()
        mixed_batch_y = batch_y.clone()

        for i in range(batch_x.size(0)):
            label_ids = batch_y[i].nonzero(as_tuple=True)[0]

            n = min(target_num, label_ids.size(0))
            selected_label_ids = np.random.choice(label_ids.cpu(), n, replace=False)

            for label_id in selected_label_ids:
                neigh = adj[label_id].indices
                n = min(mixup_num - 1, len(neigh))
                selected_neigh = np.random.choice(
                    neigh,
                    size=n,
                    p=torch.tensor(inv_w[neigh]).softmax(dim=-1),
                    replace=False,
                )

                lamda = (
                    torch.distributions.Dirichlet(torch.tensor([mixup_alpha] * (n + 1)))
                    .sample()
                    .to(device)
                    .to(outputs.dtype)
                )

                mixed = torch.cat(
                    [
                        mixed_outputs[i, label_id].unsqueeze(0),
                        mixed_outputs[i, selected_neigh],
                    ],
                    dim=0,
                )
                mixed = lamda.unsqueeze(1) * mixed

                mixup_label_ids = [label_id] + selected_neigh.tolist()
                mixed_outputs[i, mixup_label_ids] = mixed

                if not no_label_smoothing:
                    mixed_batch_y[i, mixup_label_ids] = lamda.float()
                else:
                    mixed_batch_y[i, mixup_label_ids] = 1.0

        if flow_mixup_enabled:
            outputs = model((outputs, *others), **output_opts)[0]
            loss = criterion(outputs, batch_y)
        else:
            loss = 0

        mixed_outputs = model((mixed_outputs, *others), **output_opts)[0]
        loss = loss + criterion(mixed_outputs, mixed_batch_y)

    optim.zero_grad()
    scaler.scale(loss).backward()
    clip_gradient(model, gradient_norm_queue, gradient_clip_value)
    scaler.step(optim)
    scaler.update()

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
) -> Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    models = [model]
    lr_list = [lr]
    decay_list = [decay]

    param_groups = [
        (
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": decay,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        )
        for model, lr, decay in zip(models, lr_list, decay_list)
    ]

    param_groups = [p for m_param_groups in param_groups for p in m_param_groups]

    return DenseSparseAdamW(param_groups)


@click.command(context_settings={"show_default": True})
@click.option(
    "--mode",
    type=click.Choice(["train", "eval"]),
    default="train",
    help="train: train and eval are executed. eval: eval only",
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
@click.option(
    "--mixup-enabled",
    is_flag=True,
    default=False,
    help="Enable mixup",
)
@click.option(
    "--mixup-alpha", type=click.FLOAT, default=0.4, help="Hyper parameter for mixup"
)
@click.option(
    "--mixup-warmup",
    type=click.INT,
    default=20,
    help="Deferred stragtegy for mixup. Disable: -1",
)
@click.option("--mixup-num", type=click.INT, default=2, help="# of samples to be mixed")
@click.option(
    "--in-place-enabled", is_flag=True, default=False, help="Enable in-place mixup"
)
@click.option(
    "--in-place-ver",
    type=click.IntRange(1, 2),
    default=1,
    help="version of in-place mixup",
)
@click.option(
    "--in-place-target-num",
    type=click.INT,
    default=3,
    help="# of target for in-place mixup v2",
)
@click.option(
    "--flow-mixup-enabled", is_flag=True, default=False, help="Enable flow mixup"
)
@click.option(
    "--no-label-smoothing", is_flag=True, default=False, help="No label smoothing"
)
@log_elapsed_time
def main(
    mode: str,
    test_run: bool,
    run_script: str,
    seed: int,
    model_cnf: str,
    data_cnf: str,
    ckpt_root_path: str,
    ckpt_name: str,
    mp_enabled: bool,
    swa_warmup: int,
    eval_step: int,
    print_step: int,
    early: int,
    early_criterion: str,
    num_epochs: int,
    train_batch_size: int,
    test_batch_size: int,
    no_cuda: bool,
    num_workers: int,
    decay: float,
    lr: float,
    enable_loss_weight: bool,
    resume: bool,
    resume_ckpt_path: str,
    gradient_max_norm: float,
    mixup_enabled: bool,
    mixup_alpha: float,
    mixup_warmup: int,
    mixup_num: int,
    in_place_enabled: bool,
    in_place_ver: int,
    in_place_target_num: int,
    flow_mixup_enabled: bool,
    no_label_smoothing: bool,
):
    ################################ Assert options ##################################
    ##################################################################################

    ################################ Initialize Config ###############################
    yaml = YAML(typ="safe")

    model_cnf_path = model_cnf
    data_cnf_path = data_cnf

    model_cnf = yaml.load(Path(model_cnf))
    data_cnf = yaml.load(Path(data_cnf))

    model_name = model_cnf["name"]
    dataset_name = data_cnf["name"]

    prefix = "" if ckpt_name is None else f"{ckpt_name}_"
    ckpt_name = f"{prefix}{model_name}_{dataset_name}_{seed}"
    ckpt_root_path = os.path.join(ckpt_root_path, ckpt_name)
    ckpt_path = os.path.join(ckpt_root_path, "ckpt.pt")
    last_ckpt_path = os.path.join(ckpt_root_path, "ckpt.last.pt")
    log_filename = "train.log"

    os.makedirs(ckpt_root_path, exist_ok=True)

    if not resume and os.path.exists(ckpt_path) and mode == "train":
        click.confirm(
            "Checkpoint is already existed. Overwrite it?", abort=True, err=True
        )
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

    if in_place_ver == 2:
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
    optimizer = get_optimizer(model, lr, decay)

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
                optimizer,
                scaler,
                scheduler,
                set_rng_state=True,
                return_other_states=True,
            )

            start_epoch += 1
            epoch = start_epoch
            global_step = ckpt["global_step"]
            gradient_norm_queue = ckpt["gradient_norm_queue"]
            model_swa_state = ckpt["model_swa_state"]
            best = ckpt["best"]
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
    if mixup_enabled:
        mixup_fn = MixUp(mixup_alpha)
    else:
        mixup_fn = None

    input_opts = model_cnf["train"]["input_opts"]
    output_opts = model_cnf["train"]["output_opts"]

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

                for i, (batch_x, batch_y) in enumerate(train_dataloader, 1):
                    if epoch >= mixup_warmup and mixup_enabled:
                        if in_place_enabled:
                            if in_place_ver == 1:
                                train_loss = train_in_place_mixup_step(
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
                                train_loss = train_in_place_mixup_step_v2(
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
                                    no_label_smoothing,
                                    input_opts,
                                    output_opts,
                                    inv_w,
                                    gradient_max_norm,
                                    gradient_norm_queue,
                                    device,
                                )
                        else:
                            train_loss = train_mixup_step(
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
                        train_loss = train_step(
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

                    if scheduler is not None:
                        scheduler.step()

                    train_losses.append(train_loss)

                    global_step += 1

                    val_log_msg = ""
                    if global_step % eval_step == 0 or (
                        epoch == num_epochs - 1 and i == len(train_dataloader)
                    ):
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

                        swa_step(model, model_swa_state)
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
        f"\npsn@1,3,5,10: {results['psn1']:.4f}, {results['psn3']:.4f}, {results['psn5']:.4f}, {results['psn10']:.4f}"
        # f"\nr@1,5,10: {results['r1']:.4f}, {results['r5']:.4f}, {results['r10']:.4f}"
        f"\navg.rank: {results['avg.rank']:.2f} "
        f"\ncoverage: {results['coverage']:.2f}"
    )
    logger.info(f"checkpoint name: {os.path.basename(ckpt_name)}")
    ##################################################################################


if __name__ == "__main__":
    main()
