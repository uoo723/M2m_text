"""
Created on 2021/07/06
@author Sangwoo Han

Instace Anchor + Dynamic Cluster Assignements
"""
import os
import shutil
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger
from ruamel.yaml import YAML
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from m2m_text.cluster import get_clusters
from m2m_text.datasets import AmazonCat13K, EURLex4K, Wiki10, Wiki10_31K
from m2m_text.datasets.custom import IDDataset
from m2m_text.datasets.sbert import SBertDataset, collate_fn
from m2m_text.datasets.text import TextDataset
from m2m_text.loss import CircleLoss, CircleLoss2, CircleLoss3
from m2m_text.metrics import (
    get_inv_propensity,
    get_n_1,
    get_n_3,
    get_n_5,
    get_p_1,
    get_p_3,
    get_p_5,
    get_psndcg_1,
    get_psndcg_3,
    get_psndcg_5,
    get_psp_1,
    get_psp_3,
    get_psp_5,
    get_r_1,
    get_r_5,
    get_r_10,
)
from m2m_text.networks import (
    AttentionRNN2,
    AttentionRNNEncoder,
    AttentionRNNEncoder2,
    LabelEncoder,
    RNNEncoder,
    SBert,
)
from m2m_text.optimizers import DenseSparseAdamW
from m2m_text.utils.data import copy_file, get_mlb
from m2m_text.utils.model import freeze_model, load_checkpoint2, save_checkpoint2
from m2m_text.utils.train import (
    clip_gradient,
    get_embeddings,
    get_label_embeddings,
    log_elapsed_time,
    set_logger,
    set_seed,
    swa_init,
    swa_step,
    swap_swa_params,
    to_device,
)

DATASET_CLS = {
    "AmazonCat13K": AmazonCat13K,
    "EURLex4K": EURLex4K,
    "Wiki10": Wiki10,
    "Wiki10_31K": Wiki10_31K,
}

BASE_ENCODER_CLS = {"RNNEncoder": RNNEncoder}
MATCHER_CLS = {"AttentionRNN2": AttentionRNN2}
ENCODER_CLS = {"AttentionRNNEncoder2": AttentionRNNEncoder2}
LE_MODEL_CLS = {"LabelEncoder": LabelEncoder}

TRANSFORMER_MODELS = ["SBert"]


class Collector:
    def __init__(
        self,
        dataset: IDDataset[Subset[TextDataset]],
        cluster_y: csr_matrix,
    ) -> None:
        self.dataset = dataset
        self._cluster_y = cluster_y[dataset.dataset.indices]

    @property
    def cluster_y(self) -> csr_matrix:
        return self._cluster_y

    @cluster_y.setter
    def cluster_y(self, value: csr_matrix) -> None:
        self._cluster_y = value[self.dataset.dataset.indices]

    def __call__(
        self, batch: Iterable[Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.stack([b[0] for b in batch])
        batch_x = torch.stack([b[1] for b in batch])
        batch_y = torch.stack([b[2] for b in batch])
        batch_cluster_y = torch.from_numpy(
            self._cluster_y[input_ids].toarray().squeeze()
        ).float()

        return input_ids, batch_x, batch_y, batch_cluster_y


def pool_initializer(
    _pos_num_labels: int,
    _neg_num_labels: int,
    _weight_pos_sampling: bool,
    _label_embeddings: torch.Tensor,
) -> None:
    global pos_num_labels, neg_num_labels, weight_pos_sampling, label_embeddings
    pos_num_labels = _pos_num_labels
    neg_num_labels = _neg_num_labels
    weight_pos_sampling = _weight_pos_sampling
    label_embeddings = _label_embeddings


def sample_pos_neg_single(
    emb: torch.Tensor,
    pos: torch.Tensor,
    candidate_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global pos_num_labels, neg_num_labels, weight_pos_sampling, label_embeddings

    if weight_pos_sampling:
        sim = (
            F.normalize(emb[None, ...], dim=-1)
            @ F.normalize(label_embeddings[pos], dim=-1).T
        )[0]
        p = (1 - sim).numpy()
        p /= p.sum()

    pos = torch.from_numpy(
        np.random.choice(
            pos, size=(pos_num_labels,), replace=len(pos) < pos_num_labels, p=p
        )
    ).long()

    neg = []
    sim = (
        F.normalize(emb[None, ...], dim=-1)
        @ F.normalize(label_embeddings[candidate_labels], dim=-1).T
    )[0]

    sorted_idx = sim.argsort(descending=True)

    for idx in sorted_idx:
        if len(neg) >= neg_num_labels:
            break

        if candidate_labels[idx] not in pos:
            neg.append(candidate_labels[idx])

    assert len(neg) == neg_num_labels
    neg = torch.tensor(neg, dtype=torch.long)

    return pos, neg


# Bottleneck
def sample_pos_neg(
    inputs: torch.Tensor,
    batch_y: torch.Tensor,
    clusters: torch.Tensor,
    label_embeddings: torch.Tensor,
    cluster_to_label: csr_matrix,
    pos_num_labels: int,
    neg_num_labels: int,
    pool: mp.Pool,
    weight_pos_sampling: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # pos_list = []
    # candidate_labels_list = []

    # for i, y in enumerate(batch_y):
    #     pos_list.append(y.nonzero(as_tuple=True)[0])
    #     candidate_labels_list.append(
    #         torch.from_numpy(cluster_to_label[clusters[i]].indices).long()
    #     )

    # pos, neg = zip(
    #     *pool.starmap(
    #         sample_pos_neg_single, zip(inputs, pos_list, candidate_labels_list)
    #     )
    # )

    # return torch.stack(pos), torch.stack(neg)

    positives = []
    negatives = []

    # row, col = batch_y.nonzero(as_tuple=True)

    # candidate_labels = np.concatenate([cluster_to_label[c].indices for c in clusters])
    # neg_sim = (
    #     F.normalize(inputs, dim=-1)
    #     @ F.normalize(label_embeddings[candidate_labels], dim=-1).T
    # ).cpu()

    # if weight_pos_sampling:
    #     pos_sim = (
    #         F.normalize(inputs, dim=-1) @ F.normalize(label_embeddings[col], dim=-1).T
    #     ).cpu()
    # else:
    #     pos_sim = None

    # start = 0
    # end = 0

    # for i, c in enumerate(clusters):
    #     pos = col[row == i]
    #     if pos_sim is not None:
    #         sim = pos_sim[i, row == i]
    #         p = (1 - sim).numpy()
    #         p /= p.sum()
    #     else:
    #         p = None

    #     positives.append(
    #         torch.from_numpy(
    #             np.random.choice(
    #                 pos,
    #                 size=(pos_num_labels,),
    #                 replace=len(pos) < pos_num_labels,
    #                 p=p,
    #             )
    #         )
    #     )

    # for i, c in enumerate(clusters):
    #     neg = []
    #     candidate_labels = cluster_to_label[c].indices

    #     end = start + len(candidate_labels)
    #     sim = neg_sim[i, start:end]
    #     sorted_idx = sim.argsort(descending=True)

    #     for idx in sorted_idx:
    #         if len(neg) >= neg_num_labels:
    #             break

    #         if candidate_labels[idx] not in pos:
    #             neg.append(candidate_labels[idx])

    #     assert len(neg) == neg_num_labels

    #     start = end

    #     negatives.append(torch.tensor(neg, dtype=torch.int64))

    for i, y in enumerate(batch_y):
        pos = y.nonzero(as_tuple=True)[0]

        if weight_pos_sampling:
            sim = (
                F.normalize(inputs[[i]], dim=-1)
                @ F.normalize(label_embeddings[pos], dim=-1).T
            )[0]
            p = (1 - sim).numpy()
            p /= p.sum()

        positives.append(
            torch.from_numpy(
                np.random.choice(
                    pos,
                    size=(pos_num_labels,),
                    replace=len(pos) < pos_num_labels,
                    p=p,
                )
            )
        )

        neg = []
        candidate_labels = cluster_to_label[clusters[i]].indices
        sim = (
            F.normalize(inputs[[i]], dim=-1)
            @ F.normalize(label_embeddings[candidate_labels], dim=-1).T
        )[0]
        sorted_idx = sim.argsort(descending=True)

        for idx in sorted_idx:
            if len(neg) >= neg_num_labels:
                break

            if candidate_labels[idx] not in pos:
                neg.append(candidate_labels[idx])

        assert len(neg) == neg_num_labels

        negatives.append(torch.tensor(neg, dtype=torch.int64))

    return torch.stack(positives), torch.stack(negatives)


def get_model(
    model_cnf: dict,
    le_model_cnf: dict,
    data_cnf: dict,
    num_clusters: int,
    num_labels: int,
    mp_enabled: bool,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    model_name = model_cnf["name"]
    le_model_name = le_model_cnf["name"]

    model_cnf["base_encoder"]["emb_init"] = data_cnf["model"]["emb_init"]

    base_encoder = BASE_ENCODER_CLS[model_name["base_encoder"]](
        mp_enabled=mp_enabled, **model_cnf["base_encoder"]
    ).to(device)
    matcher = MATCHER_CLS[model_name["matcher"]](
        mp_enabled=mp_enabled, num_labels=num_clusters, **model_cnf["matcher"]
    ).to(device)
    encoder = ENCODER_CLS[model_name["encoder"]](
        mp_enabled=mp_enabled, **model_cnf["encoder"]
    ).to(device)
    label_encoder = LE_MODEL_CLS[le_model_name](
        num_labels=num_labels, mp_enabled=mp_enabled, **le_model_cnf["model"]
    ).to(device)

    return base_encoder, matcher, encoder, label_encoder


def get_cluster_data(
    cluster: np.ndarray, num_labels: int, datasets: Iterable[TextDataset]
) -> Tuple[
    Iterable[csr_matrix],
    Iterable[np.ndarray],
    MultiLabelBinarizer,
    csr_matrix,
    csr_matrix,
]:
    num_clusters = len(cluster)

    label_to_cluster = lil_matrix((num_labels, num_clusters), dtype=np.int64)
    for i, label_ids in enumerate(cluster):
        label_to_cluster[label_ids, i] = 1
    label_to_cluster = label_to_cluster.tocsr()
    cluster_to_label = label_to_cluster.T.tocsr()

    cluster_y_list = []
    raw_cluster_y_list = []

    for dataset in datasets:
        cluster_y = dataset.y @ label_to_cluster
        cluster_y.data[:] = 1
        cluster_y_list.append(cluster_y)

        raw_cluster_y = np.array([y.indices for y in cluster_y], dtype=np.object)
        raw_cluster_y_list.append(raw_cluster_y)

    cluster_mlb = MultiLabelBinarizer(
        classes=np.arange(num_clusters), sparse_output=True
    ).fit(raw_cluster_y_list[0])

    return (
        cluster_y_list,
        raw_cluster_y_list,
        cluster_mlb,
        label_to_cluster,
        cluster_to_label,
    )


def build_cluster(
    labels_f: Union[csr_matrix, np.ndarray],
    cluster_path: str,
    cluster_level: int,
    num_labels: int,
    datasets: Iterable[TextDataset],
    verbose: bool = True,
    force: bool = False,
) -> Tuple[
    np.ndarray,
    Iterable[csr_matrix],
    Iterable[np.ndarray],
    MultiLabelBinarizer,
    csr_matrix,
    csr_matrix,
]:
    if os.path.exists(cluster_path) and not force:
        cluster = np.load(cluster_path, allow_pickle=True)
    else:
        cluster = get_clusters(labels_f, levels=[cluster_level], verbose=verbose)[0]
        np.save(cluster_path, cluster)

    ret = get_cluster_data(cluster, num_labels, datasets)

    return (cluster, *ret)


def train_step(
    base_encoder: nn.Module,
    matcher: nn.Module,
    encoder: nn.Module,
    label_encoder: nn.Module,
    cls_criterion: nn.Module,
    metric_criterion: nn.Module,
    top_b: int,
    cluster_to_label: csr_matrix,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_cluster_y: torch.Tensor,
    label_embeddings: torch.Tensor,
    pos_num_labels: int,
    neg_num_labels: int,
    pool: mp.Pool,
    weight_pos_sampling: bool = False,
    train_encoder: bool = False,
    scaler: Optional[GradScaler] = None,
    optim: Optional[Optimizer] = None,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    inv_w: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> float:
    base_encoder.train()
    matcher.train()
    encoder.train()
    label_encoder.train()
    mp_enabled = scaler is not None

    with torch.cuda.amp.autocast(enabled=mp_enabled):
        base_outputs, masks = base_encoder(to_device(batch_x, device))
        matcher_outputs = matcher(base_outputs, masks)[0]  # N x C
        cls_loss = cls_criterion(matcher_outputs, to_device(batch_cluster_y, device))

        if train_encoder:
            enc_outputs = encoder(base_outputs, masks)[0]
            _, clusters = torch.topk(matcher_outputs, top_b)
            clusters = clusters.cpu()  # N x top_b
            pos_labels, neg_labels = sample_pos_neg(
                enc_outputs.detach().cpu().float(),
                batch_y,
                clusters,
                label_embeddings,
                cluster_to_label,
                pos_num_labels,
                neg_num_labels,
                pool,
                weight_pos_sampling=weight_pos_sampling,
            )

            pos_inv_w = inv_w[pos_labels].to(device) if inv_w is not None else None

            pos_label_outputs = label_encoder(to_device(pos_labels, device))
            neg_label_outputs = label_encoder(to_device(neg_labels, device))

            metric_loss = metric_criterion(
                enc_outputs, pos_label_outputs, neg_label_outputs, pos_inv_w
            )
        else:
            metric_loss = torch.tensor(0.0)

        loss = cls_loss + metric_loss

    optim.zero_grad()

    if mp_enabled:
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        clip_gradient(base_encoder, gradient_norm_queue, gradient_clip_value)
        clip_gradient(matcher, gradient_norm_queue, gradient_clip_value)

        if train_encoder:
            clip_gradient(encoder, gradient_norm_queue, gradient_clip_value)
            clip_gradient(label_encoder, gradient_norm_queue, gradient_clip_value)

        scaler.step(optim)

        if train_encoder:
            label_embeddings[pos_labels] = pos_label_outputs.detach().cpu().float()
            label_embeddings[neg_labels] = neg_label_outputs.detach().cpu().float()

        scaler.update()
    else:
        loss.backward()
        clip_gradient(base_encoder, gradient_norm_queue, gradient_clip_value)
        clip_gradient(matcher, gradient_norm_queue, gradient_clip_value)

        if train_encoder:
            clip_gradient(encoder, gradient_norm_queue, gradient_clip_value)
            clip_gradient(label_encoder, gradient_norm_queue, gradient_clip_value)

        optim.step()

        if train_encoder:
            label_embeddings[pos_labels] = pos_label_outputs.detach().cpu().float()
            label_embeddings[neg_labels] = neg_label_outputs.detach().cpu().float()

    return loss.item()


def predict_step(
    base_encoder: nn.Module,
    matcher: nn.Module,
    encoder: nn.Module,
    label_embeddings: torch.Tensor,
    batch_x: torch.Tensor,
    cluster_to_label: csr_matrix,
    top_b: int,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    base_encoder.eval()
    matcher.eval()
    encoder.eval()

    with torch.no_grad():
        base_outputs, masks = base_encoder(batch_x, mp_enabled=False)
        matcher_outputs = matcher(base_outputs, masks, mp_enabled=False)[0]
        enc_outputs = encoder(base_outputs, masks, mp_enabled=False)[0]

    _, clusters = torch.topk(matcher_outputs, top_b)
    clusters = clusters.cpu()
    enc_outputs = enc_outputs.cpu()

    prediction = []
    candidate_labels = np.concatenate([cluster_to_label[c].indices for c in clusters])
    all_sim = (
        F.normalize(enc_outputs, dim=-1)
        @ F.normalize(label_embeddings[candidate_labels], dim=-1).T
    )

    start = 0
    end = 0
    for i, c in enumerate(clusters):
        candidate_labels = cluster_to_label[c].indices
        end = start + len(candidate_labels)
        sim = all_sim[i, start:end]
        sorted_idx = sim.argsort(descending=True)
        prediction.append(candidate_labels[sorted_idx[:top_k]])
        start = end

    return clusters.numpy(), np.stack(prediction)


def get_results(
    base_encoder: nn.Module,
    matcher: nn.Module,
    encoder: nn.Module,
    label_embeddings: torch.Tensor,
    dataloader: DataLoader,
    raw_y: np.ndarray,
    raw_cluster_y: np.ndarray,
    cluster_to_label: csr_matrix,
    top_b: int,
    top_k: int,
    cluster_mlb: Optional[MultiLabelBinarizer] = None,
    mlb: Optional[MultiLabelBinarizer] = None,
    inv_w: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
    test_mode: bool = False,
) -> Dict[str, float]:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(raw_y)

    if cluster_mlb is None:
        cluster_mlb = MultiLabelBinarizer(sparse_output=True).fit(raw_cluster_y)

    base_encoder.eval()
    matcher.eval()
    encoder.eval()

    clusters, prediction = zip(
        *[
            predict_step(
                base_encoder,
                matcher,
                encoder,
                label_embeddings,
                to_device(batch[1], device),
                cluster_to_label,
                top_b,
                top_k,
            )
            for batch in tqdm(dataloader, leave=False)
        ]
    )
    clusters = np.concatenate(clusters)
    prediction = np.concatenate(prediction)

    clusters = cluster_mlb.classes_[clusters]
    prediction = mlb.classes_[prediction]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        c_p5 = get_p_5(clusters, raw_cluster_y, cluster_mlb)
        c_n5 = get_n_5(clusters, raw_cluster_y, cluster_mlb)
        c_r10 = get_r_10(clusters, raw_cluster_y, cluster_mlb)

        p5 = get_p_5(prediction, raw_y, mlb)
        n5 = get_n_5(prediction, raw_y, mlb)
        r10 = get_r_10(prediction, raw_y, mlb)

        results = {
            "c_p5": c_p5,
            "c_n5": c_n5,
            "c_r10": c_r10,
            "p5": p5,
            "n5": n5,
            "r10": r10,
        }

        if inv_w is not None:
            results["psp5"] = get_psp_5(prediction, raw_y, inv_w, mlb)

        if test_mode:
            c_p1 = get_p_1(clusters, raw_cluster_y, cluster_mlb)
            c_p3 = get_p_3(clusters, raw_cluster_y, cluster_mlb)
            c_n1 = get_n_1(clusters, raw_cluster_y, cluster_mlb)
            c_n3 = get_n_3(clusters, raw_cluster_y, cluster_mlb)
            c_r1 = get_r_1(clusters, raw_cluster_y, cluster_mlb)
            c_r5 = get_r_5(clusters, raw_cluster_y, cluster_mlb)

            p1 = get_p_1(prediction, raw_y, mlb)
            p3 = get_p_3(prediction, raw_y, mlb)
            n1 = get_n_1(prediction, raw_y, mlb)
            n3 = get_n_3(prediction, raw_y, mlb)
            r1 = get_r_1(prediction, raw_y, mlb)
            r5 = get_r_5(prediction, raw_y, mlb)

            others = {
                "c_p1": c_p1,
                "c_p3": c_p3,
                "c_n1": c_n1,
                "c_n3": c_n3,
                "c_r1": c_r1,
                "c_r5": c_r5,
                "p1": p1,
                "p3": p3,
                "n1": n1,
                "n3": n3,
                "r1": r1,
                "r5": r5,
            }

            if inv_w is not None:
                others["psp1"] = get_psp_1(prediction, raw_y, inv_w, mlb)
                others["psp3"] = get_psp_3(prediction, raw_y, inv_w, mlb)
                others["psn1"] = get_psndcg_1(prediction, raw_y, inv_w, mlb)
                others["psn3"] = get_psndcg_3(prediction, raw_y, inv_w, mlb)
                others["psn5"] = get_psndcg_5(prediction, raw_y, inv_w, mlb)

            results = {**results, **others}

    return results


def get_optimizer(
    base_encoder: nn.Module,
    matcher: nn.Module,
    encoder: nn.Module,
    label_encoder: nn.Module,
    base_enc_lr: float,
    base_enc_decay: float,
    matcher_lr: float,
    matcher_decay: float,
    enc_lr: float,
    enc_decay: float,
    le_lr: float,
    le_decay: float,
) -> Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    models = [base_encoder, matcher, encoder, label_encoder]
    lr_list = [base_enc_lr, matcher_lr, enc_lr, le_lr]
    decay_list = [base_enc_decay, matcher_decay, enc_decay, le_decay]

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
@click.option(
    "--le-model-cnf", type=click.Path(exists=True), help="Label Model config file path"
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
    "--matcher-early",
    type=click.INT,
    default=5,
    help="Early stop patient count for matcher",
)
@click.option(
    "--matcher-early-criterion",
    type=click.Choice(["c_p5", "c_n5"]),
    default="c_n5",
    help="Early stopping criterion for matcher",
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
    "--base-enc-decay",
    type=click.FLOAT,
    default=1e-2,
    help="Weight decay (Base Encoder)",
)
@click.option(
    "--base-enc-lr", type=click.FLOAT, default=1e-3, help="learning rate (Base Encoder)"
)
@click.option(
    "--matcher-decay", type=click.FLOAT, default=1e-2, help="Weight decay (Matcher)"
)
@click.option(
    "--matcher-lr", type=click.FLOAT, default=1e-3, help="learning rate (Matcher)"
)
@click.option(
    "--enc-decay", type=click.FLOAT, default=1e-2, help="Weight decay (Encoder)"
)
@click.option(
    "--enc-lr", type=click.FLOAT, default=1e-3, help="learning rate (Encoder)"
)
@click.option(
    "--le-decay", type=click.FLOAT, default=1e-2, help="Weight decay for label encoder"
)
@click.option(
    "--le-lr", type=click.FLOAT, default=1e-3, help="learning rate for label encoder"
)
@click.option(
    "--ann-candidates", type=click.INT, default=30, help="# of ANN candidates"
)
@click.option("--resume", is_flag=True, default=False, help="Resume training")
@click.option(
    "--pos-num-labels", type=click.INT, default=5, help="# of positive samples"
)
@click.option(
    "--neg-num-labels", type=click.INT, default=5, help="# of negative samples"
)
@click.option(
    "--loss-name",
    type=click.Choice(["circle", "circle2", "circle3"]),
    default="circle",
    help="Loss function",
)
@click.option(
    "--gradient-max-norm",
    type=click.FLOAT,
    help="max norm for gradient clipping",
)
@click.option("--m", type=click.FLOAT, default=0.15, help="Margin of Circle loss")
@click.option(
    "--gamma", type=click.FLOAT, default=1.0, help="Scale factor of Circle loss"
)
@click.option(
    "--metric",
    type=click.Choice(["cosine", "euclidean"]),
    default="cosine",
    help="metric function to be used",
)
@click.option(
    "--label-pos-neg-num",
    type=click.INT,
    nargs=3,
    default=(0, 0, 0),
    help="# of positive (negative) samples with respect to label"
    "[# of labels to sample, # of pos samples per label, # of neg samples per label]",
)
@click.option(
    "--weight-pos-sampling",
    is_flag=True,
    default=False,
    help="Enable weighted postive sampling",
)
@click.option(
    "--enable-loss-pos-weights",
    is_flag=True,
    default=False,
    help="Enable pos weights based on inv_w",
)
@click.option("--cluster-level", type=click.INT, default=10, help="Cluster level")
@click.option("--top-b", type=click.INT, default=10, help="Top b clusters")
@click.option("--top-k", type=click.INT, default=10, help="Top k labels")
@click.option(
    "--matcher-warmup", type=click.INT, default=2000, help="matcher warmup steps"
)
@click.option(
    "--building-cluster-early",
    type=click.INT,
    default=5,
    help="Build new cluster when metric is not improved",
)
@log_elapsed_time
def main(
    mode: str,
    test_run: bool,
    run_script: str,
    seed: int,
    model_cnf: str,
    le_model_cnf: str,
    data_cnf: str,
    ckpt_root_path: str,
    ckpt_name: str,
    mp_enabled: bool,
    swa_warmup: int,
    eval_step: int,
    print_step: int,
    early: int,
    early_criterion: str,
    matcher_early: int,
    matcher_early_criterion: str,
    num_epochs: int,
    train_batch_size: int,
    test_batch_size: int,
    no_cuda: bool,
    num_workers: int,
    base_enc_decay: float,
    base_enc_lr: float,
    matcher_decay: float,
    matcher_lr: float,
    enc_decay: float,
    enc_lr: float,
    le_decay: float,
    le_lr: float,
    ann_candidates: int,
    resume: bool,
    pos_num_labels: int,
    neg_num_labels: int,
    loss_name: str,
    gradient_max_norm: float,
    m: float,
    gamma: float,
    metric: str,
    label_pos_neg_num: Tuple[int, int],
    weight_pos_sampling: bool,
    enable_loss_pos_weights: bool,
    cluster_level: int,
    top_b: int,
    top_k: int,
    matcher_warmup: int,
    building_cluster_early: int,
):
    ################################ Assert options ##################################
    if loss_name != "circle3":
        assert metric == "cosine"

    if label_pos_neg_num[0] > 0:
        assert label_pos_neg_num[1] + label_pos_neg_num[2] > 0

    assert building_cluster_early < early
    ##################################################################################

    ################################ Initialize Config ###############################
    yaml = YAML(typ="safe")

    model_cnf_path = model_cnf
    le_model_cnf_path = le_model_cnf
    data_cnf_path = data_cnf

    model_cnf = yaml.load(Path(model_cnf))
    le_model_cnf = yaml.load(Path(le_model_cnf))
    data_cnf = yaml.load(Path(data_cnf))

    model_name = model_cnf["name"]
    le_model_name = le_model_cnf["name"]
    dataset_name = data_cnf["name"]

    prefix = "" if ckpt_name is None else f"{ckpt_name}_"
    ckpt_name = f"{prefix}{model_name['base_encoder']}_{dataset_name}_{seed}"
    ckpt_root_path = os.path.join(ckpt_root_path, ckpt_name)
    ckpt_path = os.path.join(ckpt_root_path, "ckpt.pt")
    last_ckpt_path = os.path.join(ckpt_root_path, "ckpt.last.pt")
    log_filename = "train.log"
    cluster_path = os.path.join(ckpt_root_path, f"cluster_{cluster_level}.npy")
    best_cluster_path = os.path.join(
        ckpt_root_path, f"best_cluster_{cluster_level}.npy"
    )

    os.makedirs(ckpt_root_path, exist_ok=True)

    if not resume and os.path.exists(ckpt_path) and mode == "train":
        click.confirm(
            "Checkpoint is already existed. Overwrite it?", abort=True, err=True
        )
        shutil.rmtree(ckpt_root_path)
        os.makedirs(ckpt_root_path, exist_ok=True)

    if not test_run:
        set_logger(os.path.join(ckpt_root_path, log_filename))

        copy_file(
            model_cnf_path,
            os.path.join(ckpt_root_path, os.path.basename(model_cnf_path)),
        )
        copy_file(
            le_model_cnf_path,
            os.path.join(ckpt_root_path, os.path.basename(le_model_cnf_path)),
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_dataset = DATASET_CLS[dataset_name](
            **data_cnf["dataset"], **model_cnf.get("dataset", {})
        )
        test_dataset = DATASET_CLS[dataset_name](
            train=False, **data_cnf["dataset"], **model_cnf.get("dataset", {})
        )
        inv_w = get_inv_propensity(train_dataset.y)
        inv_w_tensor = torch.from_numpy(inv_w)
        inv_w_tensor = inv_w_tensor / inv_w_tensor.max()

        mlb = get_mlb(train_dataset.le_path)
        num_labels = train_dataset.y.shape[1]

    train_ids = np.arange(len(train_dataset))
    train_ids, valid_ids = train_test_split(
        train_ids, test_size=data_cnf.get("valid_size", 200)
    )
    train_mask = np.zeros(len(train_dataset), dtype=np.bool)
    train_mask[train_ids] = True
    train_mask = torch.from_numpy(train_mask)

    logger.info(
        f"# of train dataset: {train_mask.nonzero(as_tuple=True)[0].shape[0]:,}"
    )
    logger.info(
        f"# of valid dataset: {(~train_mask).nonzero(as_tuple=True)[0].shape[0]:,}"
    )
    logger.info(f"# of test dataset: {len(test_dataset):,}")
    logger.info(f"# of labels: {num_labels:,}")
    ##################################################################################

    #################################### Clustering ##################################
    logger.info("Initialize Cluster")
    sparse_x = train_dataset.get_sparse_features()
    sparse_y = train_dataset.y
    labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))

    (
        cluster,
        (train_cluster_y, test_cluster_y),
        (train_raw_cluster_y, test_raw_cluster_y),
        cluster_mlb,
        label_to_cluster,
        cluster_to_label,
    ) = build_cluster(
        labels_f, cluster_path, cluster_level, num_labels, [train_dataset, test_dataset]
    )

    num_clusters = len(cluster)
    avg_num_of_labels_per_sample = train_dataset.y.sum(axis=-1).mean()
    avg_num_of_custers = train_cluster_y.sum(axis=-1).mean()
    avg_num_of_labels_per_cluster = label_to_cluster.sum(axis=0).mean()

    logger.info(f"# of clusters: {num_clusters}")
    logger.info(f"Avg. # of labels / sample: {avg_num_of_labels_per_sample:.2f}")
    logger.info(f"Avg. # of clusters / sample: {avg_num_of_custers:.2f}")
    logger.info(f"Avg. # of labels / cluser: {avg_num_of_labels_per_cluster:.2f}")
    logger.info(
        f"Avg. # of candidate labels / inst.: {avg_num_of_labels_per_cluster * top_b:.2f}"
    )
    ##################################################################################

    ################################# Prepare Model ##################################
    logger.info(f"Base Encoder: {model_name['base_encoder']}")
    logger.info(f"Matcher: {model_name['matcher']}")
    logger.info(f"Encoder: {model_name['encoder']}")
    logger.info(f"Label Model: {le_model_name}")

    base_encoder, matcher, encoder, label_encoder = get_model(
        model_cnf, le_model_cnf, data_cnf, num_clusters, num_labels, mp_enabled, device
    )

    if num_gpus > 1 and not no_cuda:
        logger.info(f"Multi-GPU mode: {num_gpus} GPUs")
        base_encoder = nn.DataParallel(base_encoder)
        matcher = nn.DataParallel(matcher)
        encoder = nn.DataParallel(encoder)
        label_encoder = nn.DataParallel(label_encoder)
    elif not no_cuda:
        logger.info("Single-GPU mode")
    else:
        logger.info("CPU mode")
    ##################################################################################

    ############################### Prepare Training #################################
    optimizer = get_optimizer(
        base_encoder,
        matcher,
        encoder,
        label_encoder,
        base_enc_lr,
        base_enc_decay,
        matcher_lr,
        matcher_decay,
        enc_lr,
        enc_decay,
        le_lr,
        le_decay,
    )

    scheduler = None
    scaler = GradScaler() if mp_enabled else None

    if loss_name == "circle":
        metric_criterion = CircleLoss(m=m, gamma=gamma)
    elif loss_name == "circle2":
        metric_criterion = CircleLoss2(m=m, gamma=gamma)
    else:
        metric_criterion = CircleLoss3(m=m, gamma=gamma, metric=metric)

    cls_criterion = nn.BCEWithLogitsLoss()

    gradient_norm_queue = (
        deque([np.inf], maxlen=5) if gradient_max_norm is not None else None
    )

    base_enc_swa_state = {}
    matcher_swa_state = {}
    enc_swa_state = {}
    le_swa_state = {}
    results = {}

    start_epoch = 0
    global_step = 0
    best, e = 0, 0
    matcher_best, matcher_e = 0, 0
    building_cluster_e = 0
    early_stop = False

    train_encoder = False

    train_losses = deque(maxlen=print_step)

    # global_matcher_warmup = matcher_warmup

    if resume and mode == "train":
        resume_ckpt_path = (
            last_ckpt_path if os.path.exists(last_ckpt_path) else ckpt_path
        )
        if os.path.exists(resume_ckpt_path):
            logger.info("Resume Training")
            start_epoch, ckpt = load_checkpoint2(
                resume_ckpt_path,
                [base_encoder, matcher, encoder, label_encoder],
                optimizer,
                scaler,
                scheduler,
                set_rng_state=True,
                return_other_states=True,
            )

            start_epoch += 1
            epoch = start_epoch
            global_step = ckpt["global_step"]
            # global_matcher_warmup = ckpt["global_matcher_warmup"]
            gradient_norm_queue = ckpt["gradient_norm_queue"]
            base_enc_swa_state = ckpt["base_enc_swa_state"]
            matcher_swa_state = ckpt["matcher_swa_state"]
            enc_swa_state = ckpt["enc_swa_state"]
            le_swa_state = ckpt["le_swa_state"]
            best = ckpt["best"]
            e = ckpt["e"]
            matcher_e = ckpt["matcher_e"]
            matcher_best = ckpt["matcher_best"]
            train_encoder = ckpt["train_encoder"]
            building_cluster_e = ckpt["building_cluster_e"]

        else:
            logger.warning("No checkpoint")
    ##################################################################################

    ############################### Prepare Dataloader ###############################
    logger.info(f"Prepare Dataloader")

    # Not contioned
    if model_name in TRANSFORMER_MODELS:
        train_texts = train_dataset.raw_data()[0]
        test_texts = test_dataset.raw_data()[0]

        tokenizer = (
            base_encoder.module.tokenize
            if isinstance(base_encoder, nn.DataParallel)
            else base_encoder.tokenize
        )

        train_sbert_dataset = SBertDataset(
            tokenizer(train_texts[train_mask]),
            train_dataset.y[train_mask],
        )
        valid_sbert_dataset = SBertDataset(
            tokenizer(train_texts[~train_mask]),
            train_dataset.y[~train_mask],
        )
        test_sbert_dataset = SBertDataset(
            tokenizer(test_texts),
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
        train_subset_dataset = IDDataset(Subset(train_dataset, train_ids))
        valid_subset_dataset = IDDataset(Subset(train_dataset, valid_ids))
        test_subset_dataset = IDDataset(
            Subset(test_dataset, np.arange(len(test_dataset)))
        )

        train_collector = Collector(train_subset_dataset, train_cluster_y)
        valid_collector = Collector(valid_subset_dataset, train_cluster_y)
        test_collector = Collector(test_subset_dataset, test_cluster_y)

        train_dataloader = DataLoader(
            train_subset_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collector,
            pin_memory=False if no_cuda else True,
        )
        valid_dataloader = DataLoader(
            valid_subset_dataset,
            batch_size=test_batch_size,
            num_workers=num_workers,
            collate_fn=valid_collector,
            pin_memory=False if no_cuda else True,
        )
        test_dataloader = DataLoader(
            test_subset_dataset,
            batch_size=test_batch_size,
            num_workers=num_workers,
            collate_fn=test_collector,
            pin_memory=False if no_cuda else True,
        )
    ##################################################################################

    logger.info(f"checkpoint name: {os.path.basename(ckpt_name)}")

    ##################################### Training ###################################
    label_embeddings = get_label_embeddings(
        label_encoder, device=device, return_pt=True
    )

    # print(mp.get_start_method())
    # mp.set_start_method("spawn")

    # pool = mp.Pool(
    #     # processes=5,
    #     initializer=pool_initializer,
    #     initargs=(
    #         pos_num_labels,
    #         neg_num_labels,
    #         weight_pos_sampling,
    #         label_embeddings,
    #     ),
    # )
    pool = None

    start_train_encoder = False

    if mode == "train":
        try:
            for epoch in range(start_epoch, num_epochs):
                if early_stop:
                    break

                for i, (
                    _,
                    batch_x,
                    batch_y,
                    batch_cluster_y,
                ) in enumerate(train_dataloader, 1):
                    if start_train_encoder:
                        logger.info("Train Encoder")
                        start_train_encoder = False

                    if train_encoder and building_cluster_e >= building_cluster_early:
                        labels_f = get_label_embeddings(label_encoder, device=device)
                        (
                            cluster,
                            (train_cluster_y, test_cluster_y),
                            (train_raw_cluster_y, test_raw_cluster_y),
                            cluster_mlb,
                            label_to_cluster,
                            cluster_to_label,
                        ) = build_cluster(
                            labels_f,
                            cluster_path,
                            cluster_level,
                            num_labels,
                            [train_dataset, test_dataset],
                            force=True,
                        )

                        train_collector.cluster_y = train_cluster_y
                        valid_collector.cluster_y = train_cluster_y
                        test_collector.cluster_y = test_cluster_y

                        # global_matcher_warmup = global_step + matcher_warmup
                        train_encoder = False
                        building_cluster_e = 0
                        matcher_e = 0
                        matcher_best = 0
                        freeze_model(base_encoder)

                    train_loss = train_step(
                        base_encoder,
                        matcher,
                        encoder,
                        label_encoder,
                        cls_criterion,
                        metric_criterion,
                        top_b,
                        cluster_to_label,
                        batch_x,
                        batch_y,
                        batch_cluster_y,
                        label_embeddings,
                        pos_num_labels,
                        neg_num_labels,
                        pool,
                        weight_pos_sampling,
                        train_encoder,
                        scaler,
                        optimizer,
                        gradient_clip_value=gradient_max_norm,
                        gradient_norm_queue=gradient_norm_queue,
                        inv_w=inv_w_tensor if enable_loss_pos_weights else None,
                        device=device,
                    )

                    if scheduler is not None:
                        scheduler.step()

                    train_losses.append(train_loss)

                    global_step += 1

                    if global_step == swa_warmup:
                        logger.info("Initialze SWA")
                        swa_init(base_encoder, base_enc_swa_state)
                        swa_init(matcher, matcher_swa_state)
                        swa_init(encoder, enc_swa_state)
                        swa_init(label_encoder, le_swa_state)

                    val_log_msg = ""
                    if global_step % eval_step == 0 or (
                        epoch == num_epochs - 1 and i == len(train_dataloader)
                    ):
                        # label_embeddings[:] = get_label_embeddings(
                        #     label_encoder, device=device, return_pt=True
                        # )
                        results = get_results(
                            base_encoder,
                            matcher,
                            encoder,
                            label_embeddings,
                            valid_dataloader,
                            train_dataset.raw_y[valid_ids],
                            train_raw_cluster_y[valid_ids],
                            cluster_to_label,
                            top_b,
                            top_k,
                            cluster_mlb,
                            mlb=mlb,
                            inv_w=inv_w,
                            device=device,
                        )

                        val_log_msg = f"\nc_p@5: {results['c_p5']:.5f} c_n@5: {results['c_n5']:.5f}"

                        if train_encoder:
                            val_log_msg += (
                                f"\np@5: {results['p5']:.5f} n@5: {results['n5']:.5f} "
                            )

                            if "psp5" in results:
                                val_log_msg += f"psp@5: {results['psp5']:.5f}"

                            if best < results[early_criterion]:
                                best = results[early_criterion]
                                e = 0
                                building_cluster_e = 0

                                save_checkpoint2(
                                    ckpt_path,
                                    epoch,
                                    [base_encoder, matcher, encoder, label_encoder],
                                    optim=optimizer,
                                    scaler=scaler,
                                    scheduler=scheduler,
                                    results=results,
                                    other_states={
                                        "best": best,
                                        "train_mask": train_mask,
                                        "train_ids": train_ids,
                                        "base_enc_swa_state": base_enc_swa_state,
                                        "matcher_swa_state": matcher_swa_state,
                                        "enc_swa_state": enc_swa_state,
                                        "le_swa_state": le_swa_state,
                                        "global_step": global_step,
                                        # "global_matcher_warmup": global_matcher_warmup,
                                        "early_criterion": early_criterion,
                                        "gradient_norm_queue": gradient_norm_queue,
                                        "e": e,
                                        "matcher_e": matcher_e,
                                        "matcher_best": matcher_best,
                                        "train_encoder": train_encoder,
                                        "building_cluster_e": building_cluster_e,
                                    },
                                )
                                shutil.copyfile(cluster_path, best_cluster_path)
                            else:
                                e += 1
                                building_cluster_e += 1

                        else:
                            if matcher_best < results[matcher_early_criterion]:
                                matcher_best = results[matcher_early_criterion]
                                matcher_e = 0
                            else:
                                matcher_e += 1

                            if matcher_e >= matcher_early:
                                train_encoder = True
                                start_train_encoder = True
                                freeze_model(base_encoder, False)

                        swa_step(base_encoder, base_enc_swa_state)
                        swa_step(matcher, matcher_swa_state)
                        swa_step(encoder, enc_swa_state)
                        swa_step(label_encoder, le_swa_state)
                        swap_swa_params(base_encoder, base_enc_swa_state)
                        swap_swa_params(matcher, matcher_swa_state)
                        swap_swa_params(encoder, enc_swa_state)
                        swap_swa_params(label_encoder, le_swa_state)

                    if (
                        global_step % print_step == 0  # print step
                        or global_step % eval_step == 0  # eval step
                        or (
                            epoch == num_epochs - 1 and i == len(train_dataloader)
                        )  # last step
                    ):
                        log_msg = f"{epoch} {i * train_dataloader.batch_size} "
                        log_msg += f"early stop: {e}/{early} "
                        log_msg += f"matcher warmup: {matcher_e}/{matcher_early} "
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
            [base_encoder, matcher, encoder, label_encoder],
            optim=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            results=results,
            other_states={
                "best": best,
                "train_mask": train_mask,
                "train_ids": train_ids,
                "base_enc_swa_state": base_enc_swa_state,
                "matcher_swa_state": matcher_swa_state,
                "enc_swa_state": enc_swa_state,
                "le_swa_state": le_swa_state,
                "global_step": global_step,
                # "global_matcher_warmup": global_matcher_warmup,
                "early_criterion": early_criterion,
                "gradient_norm_queue": gradient_norm_queue,
                "e": e,
                "matcher_e": matcher_e,
                "matcher_best": matcher_best,
                "train_encoder": train_encoder,
                "building_cluster_e": building_cluster_e,
            },
        )
    ##################################################################################

    ################################### Evaluation ###################################
    logger.info("Evaluation.")
    if os.path.exists(ckpt_path):
        load_checkpoint2(
            ckpt_path,
            [base_encoder, matcher, encoder, label_encoder],
            set_rng_state=False,
        )

    if os.path.exists(best_cluster_path):
        cluster = np.load(best_cluster_path, allow_pickle=True)

        (
            (train_cluster_y, test_cluster_y),
            (train_raw_cluster_y, test_raw_cluster_y),
            cluster_mlb,
            label_to_cluster,
            cluster_to_label,
        ) = get_cluster_data(
            cluster,
            num_labels,
            [train_dataset, test_dataset],
        )

    results = get_results(
        base_encoder,
        matcher,
        encoder,
        label_embeddings,
        test_dataloader,
        test_dataset.raw_y,
        test_raw_cluster_y,
        cluster_to_label,
        top_b,
        top_k,
        cluster_mlb,
        mlb,
        inv_w,
        device,
        test_mode=True,
    )
    logger.info(
        f"\nc_p@1,3,5: {results['c_p1']:.4f}, {results['c_p3']:.4f}, {results['c_p5']:.4f}"
        f"\nc_n@1,3,5: {results['c_n1']:.4f}, {results['c_n3']:.4f}, {results['c_n5']:.4f}"
        f"\nc_r@1,5,10: {results['c_r1']:.4f}, {results['c_r5']:.4f}, {results['c_r10']:.4f}"
        f"\np@1,3,5: {results['p1']:.4f}, {results['p3']:.4f}, {results['p5']:.4f}"
        f"\nn@1,3,5: {results['n1']:.4f}, {results['n3']:.4f}, {results['n5']:.4f}"
        f"\npsp@1,3,5: {results['psp1']:.4f}, {results['psp3']:.4f}, {results['psp5']:.4f}"
        f"\npsn@1,3,5: {results['psn1']:.4f}, {results['psn3']:.4f}, {results['psn5']:.4f}"
        f"\nr@1,5,10: {results['r1']:.4f}, {results['r5']:.4f}, {results['r10']:.4f}"
    )
    logger.info(f"checkpoint name: {os.path.basename(ckpt_name)}")
    ##################################################################################


if __name__ == "__main__":
    main()
