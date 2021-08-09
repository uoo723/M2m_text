"""
Created on 2021/07/06
@author Sangwoo Han

Instance Anchor

"""
import multiprocessing
import os
import random
import shutil
import time
import warnings
from collections import deque
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import click
import logzero
import numpy as np
import torch
import torch.nn as nn
from logzero import logger
from ruamel.yaml import YAML
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from m2m_text.anns import HNSW, build_ann, build_ann_async, copy_ann_index, load_ann
from m2m_text.datasets import AmazonCat13K, EURLex4K, Wiki10, Wiki10_31K
from m2m_text.datasets.custom import IDDataset
from m2m_text.datasets.sbert import SBertDataset, collate_fn
from m2m_text.datasets.text import TextDataset
from m2m_text.loss import CircleLoss, CircleLoss2, CircleLoss3
from m2m_text.metrics import (
    get_inv_propensity,
    get_n_5,
    get_p_5,
    get_precision_results,
    get_psp_5,
    get_r_10,
)
from m2m_text.networks import AttentionRNNEncoder, LabelEncoder, SBert
from m2m_text.optimizers import DenseSparseAdamW
from m2m_text.utils.data import copy_file, get_mlb
from m2m_text.utils.model import load_checkpoint, save_checkpoint
from m2m_text.utils.train import (
    clip_gradient,
    get_embeddings,
    get_label_embeddings,
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

MODEL_CLS = {
    "AttentionRNNEncoder": AttentionRNNEncoder,
    "SBert": SBert,
}

LE_MODEL_CLS = {
    "LabelEncoder": LabelEncoder,
}

TRANSFORMER_MODELS = ["SBert"]


class Collector:
    def __init__(
        self,
        dataset: IDDataset[Subset[TextDataset]],
        ann_index: HNSW,
        pos_num_labels: int = 5,
        neg_num_labels: int = 5,
        label_pos_neg_num: Tuple[int, int, int] = (0, 0, 0),
        weight_pos_sampling: bool = False,
    ) -> None:
        self.dataset = dataset
        self.y = dataset.dataset.dataset.y[dataset.dataset.indices]
        self.inverted_index = self.y.T.tocsr()

        self.ann_index = ann_index
        self.pos_num_labels = pos_num_labels
        self.neg_num_labels = neg_num_labels
        self.label_pos_neg_num = label_pos_neg_num
        self.weight_pos_sampling = weight_pos_sampling

    def __call__(
        self, batch: Iterable[Tuple[torch.Tensor, ...]]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        anchor_doc_ids = np.array([b[0].item() for b in batch])
        batch_y = np.stack([b[2].numpy() for b in batch])
        anchor_doc_emb = self.ann_index.input_embeddings[anchor_doc_ids]
        neigh_labels = self.ann_index.kneighbors(anchor_doc_emb, return_distance=False)

        pos_labels = []
        neg_labels = []
        total_pos_labels = []

        ################################ Anchor: document #############################
        for i, y in enumerate(batch_y):
            pos = y.nonzero()[0]
            total_pos_labels.append(pos)

            if self.weight_pos_sampling:
                sim = (
                    normalize(anchor_doc_emb[[i]])
                    @ normalize(self.ann_index.embeddings[pos]).T
                )[0]
                p = 1 - sim
                p /= p.sum()
            else:
                p = None

            pos_labels.append(
                np.random.choice(
                    pos,
                    size=(self.pos_num_labels,),
                    replace=len(pos) < self.pos_num_labels,
                    p=p,
                )
            )

            neg = []
            for neigh_label in neigh_labels[i]:
                if len(neg) >= self.neg_num_labels:
                    break

                if neigh_label != -1 and neigh_label not in pos:
                    neg.append(neigh_label)

            assert len(neg) == self.neg_num_labels
            neg_labels.append(np.array(neg, dtype=np.int64))

        anchor_doc_ids = torch.from_numpy(anchor_doc_ids)
        anchor_doc = self.dataset[anchor_doc_ids][1]
        pos_labels = torch.from_numpy(np.stack(pos_labels))
        neg_labels = torch.from_numpy(np.stack(neg_labels))

        total_pos_labels = np.unique(np.concatenate(total_pos_labels))
        ###############################################################################

        ################################# Anchor: label ###############################
        pos_inst = []
        neg_inst = []

        if self.label_pos_neg_num[0] > 0:
            anchor_label_ids = np.random.choice(
                total_pos_labels, size=(self.label_pos_neg_num[0],), replace=False
            )

            doc_ids = np.unique(self.inverted_index[anchor_label_ids].indices)

            for i, label_id in enumerate(anchor_label_ids):
                pos_doc_ids = self.inverted_index[label_id].indices
                sample_pos_doc_ids = np.random.choice(
                    pos_doc_ids,
                    size=(self.label_pos_neg_num[1],),
                    replace=len(pos_doc_ids) < self.label_pos_neg_num[1],
                )

                if len(sample_pos_doc_ids) > 0:
                    pos_inst.append(sample_pos_doc_ids)

                sample_neg_doc_ids = []
                while len(sample_neg_doc_ids) < self.label_pos_neg_num[2]:
                    doc_id = doc_ids[np.random.randint(doc_ids.shape[0], size=1)[0]]

                    if doc_id not in pos_doc_ids and doc_id not in sample_neg_doc_ids:
                        sample_neg_doc_ids.append(doc_id)

                if len(sample_neg_doc_ids) > 0:
                    neg_inst.append(np.array(sample_neg_doc_ids))

            anchor_label_ids = torch.from_numpy(anchor_label_ids)

            if len(pos_inst) > 0:
                pos_inst = np.stack(pos_inst)
                pos_inst = self.dataset[pos_inst.ravel()][1].view(*pos_inst.shape, -1)
            else:
                pos_inst = None

            if len(neg_inst) > 0:
                neg_inst = np.stack(neg_inst)
                neg_inst = self.dataset[neg_inst.ravel()][1].view(*neg_inst.shape, -1)
            else:
                neg_inst = None
        else:
            anchor_label_ids = None
            pos_inst = None
            neg_inst = None
        ###############################################################################

        return (
            (anchor_doc_ids, anchor_doc, pos_labels, neg_labels),
            (anchor_label_ids, pos_inst, neg_inst),
        )


def log_elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        logger.info(f"elapsed time: {end - start:.2f}s, {timedelta(seconds=elapsed)}")

        return ret

    return wrapper


def set_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logzero.logfile(log_path)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


def get_model(
    model_name: str,
    model_cnf: dict,
    data_cnf: dict,
    mp_enabled: bool,
    device: torch.device,
) -> nn.Module:
    if model_name in TRANSFORMER_MODELS:
        model = MODEL_CLS[model_name](mp_enabled=mp_enabled, **model_cnf["model"]).to(
            device
        )
    else:
        model = MODEL_CLS[model_name](
            mp_enabled=mp_enabled, **model_cnf["model"], **data_cnf["model"]
        ).to(device)

    return model


def train_step(
    model: nn.Module,
    label_encoder: nn.Module,
    criterion: nn.Module,
    ann_index: HNSW,
    batch_anchor_doc_ids: torch.Tensor,
    batch_anchor_doc: torch.Tensor,
    batch_pos_labels: torch.Tensor,
    batch_neg_labels: torch.Tensor,
    batch_anchor_label_ids: Optional[torch.Tensor] = None,
    batch_pos_inst: Optional[torch.Tensor] = None,
    batch_neg_inst: Optional[torch.Tensor] = None,
    scaler: Optional[GradScaler] = None,
    optim: Optional[Optimizer] = None,
    is_train: bool = True,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
    inv_w: Optional[torch.Tensor] = None,
):
    model.train(is_train)
    label_encoder.train(is_train)
    mp_enabled = scaler is not None

    with torch.set_grad_enabled(is_train):
        with torch.cuda.amp.autocast(enabled=mp_enabled):
            pos_inv_w = (
                inv_w[batch_pos_labels].to(device) if inv_w is not None else None
            )

            anchor_doc_outputs = model(batch_anchor_doc.to(device))[0]
            pos_label_outputs = label_encoder(batch_pos_labels.to(device))
            neg_label_outputs = label_encoder(batch_neg_labels.to(device))

            ann_index.input_embeddings[batch_anchor_doc_ids] = (
                anchor_doc_outputs.detach().cpu().float().numpy()
            )

            anchor_doc_loss = criterion(
                anchor_doc_outputs, pos_label_outputs, neg_label_outputs, pos_inv_w
            )

            if batch_anchor_label_ids is not None:
                anchor_label_outputs = label_encoder(batch_anchor_label_ids.to(device))
                pos_inst_outputs = (
                    model(
                        batch_pos_inst.view(
                            batch_pos_inst.size()[:2].numel(), batch_pos_inst.size(-1)
                        ).to(device)
                    )[0].view(*batch_pos_inst.size()[:2], -1)
                    if batch_pos_inst is not None
                    else None
                )
                neg_inst_outputs = (
                    model(
                        batch_neg_inst.view(
                            batch_neg_inst.size()[:2].numel(), batch_neg_inst.size(-1)
                        ).to(device)
                    )[0].view(*batch_neg_inst.size()[:2], -1)
                    if batch_neg_inst is not None
                    else None
                )

                anchor_label_loss = criterion(
                    anchor_label_outputs, pos_inst_outputs, neg_inst_outputs
                )
            else:
                anchor_label_loss = 0.0

            loss = anchor_doc_loss + anchor_label_loss

        if is_train:
            optim.zero_grad()

            if mp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                clip_gradient(model, gradient_norm_queue, gradient_clip_value)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                clip_gradient(model, gradient_norm_queue, gradient_clip_value)
                optim.step()

    return loss.item()


def get_results(
    model: nn.Module,
    dataloader: DataLoader,
    raw_y: np.ndarray,
    ann_index: HNSW,
    mlb: Optional[MultiLabelBinarizer] = None,
    inv_w: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(raw_y)

    test_embeddings = get_embeddings(model, dataloader, device, leave=False)

    _, test_neigh = ann_index.kneighbors(test_embeddings)

    prediction = mlb.classes_[test_neigh]

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

    return results


def get_optimizer(
    model: nn.Module,
    label_encoder: nn.Module,
    lr: float,
    decay: float,
    le_lr: float,
    le_decay: float,
) -> Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
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
        {
            "params": [
                p
                for n, p in label_encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": le_decay,
            "lr": le_lr,
        },
        {
            "params": [
                p
                for n, p in label_encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": le_lr,
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
@click.option("--decay", type=click.FLOAT, default=1e-2, help="Weight decay")
@click.option("--lr", type=click.FLOAT, default=1e-5, help="learning rate")
@click.option(
    "--le-decay", type=click.FLOAT, default=1e-2, help="Weight decay for label encoder"
)
@click.option(
    "--le-lr", type=click.FLOAT, default=1e-3, help="learning rate for label encoder"
)
@click.option(
    "--ann-candidates", type=click.INT, default=30, help="# of ANN candidates"
)
@click.option(
    "--freeze-model", is_flag=True, default=False, help="Freeze model parameters"
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
    num_epochs: int,
    train_batch_size: int,
    test_batch_size: int,
    no_cuda: bool,
    num_workers: int,
    decay: float,
    lr: float,
    le_decay: float,
    le_lr: float,
    ann_candidates: int,
    freeze_model: bool,
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
):
    ################################ Assert options ##################################
    if loss_name != "circle3":
        assert metric == "cosine"

    if label_pos_neg_num[0] > 0:
        assert label_pos_neg_num[1] + label_pos_neg_num[2] > 0
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
    ckpt_name = f"{prefix}{model_name}_{dataset_name}_{seed}.pt"
    ckpt_root_path = os.path.join(ckpt_root_path, os.path.splitext(ckpt_name)[0])
    ckpt_path = os.path.join(ckpt_root_path, ckpt_name)
    last_ckpt_path = os.path.splitext(ckpt_path)
    last_ckpt_path = last_ckpt_path[0] + ".last" + last_ckpt_path[1]
    log_filename = os.path.splitext(ckpt_name)[0] + ".log"

    ann_index_filepath = os.path.join(ckpt_root_path, "ann_index")
    best_ann_index_filepath = os.path.join(ckpt_root_path, "best_ann_index")
    label_embedding_filepath = os.path.join(ckpt_root_path, "label_embeddings.npy")
    best_label_embedding_filepath = os.path.join(
        ckpt_root_path, "best_label_embeddings.npy"
    )

    os.makedirs(ckpt_root_path, exist_ok=True)

    if not resume and os.path.exists(ckpt_path) and mode == "train":
        click.confirm(
            "Checkpoint is already existed. Overwrite it?", abort=True, err=True
        )

    if not test_run:
        logfile_path = os.path.join(ckpt_root_path, log_filename)
        if os.path.exists(logfile_path) and not resume and mode == "train":
            os.remove(logfile_path)
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

    dataset_path = os.path.dirname(train_dataset.tokenized_path)

    train_tokenized_texts = np.load(
        os.path.join(dataset_path, "train_raw.npz"), allow_pickle=True
    )["texts"]
    test_tokenized_texts = np.load(
        os.path.join(dataset_path, "test_raw.npz"), allow_pickle=True
    )["texts"]

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

    ################################# Prepare Model ##################################
    logger.info(f"Model: {model_name}")
    logger.info(f"Label Model: {le_model_name}")

    model = get_model(model_name, model_cnf, data_cnf, mp_enabled, device)

    label_encoder = LE_MODEL_CLS[le_model_name](
        num_labels=num_labels, mp_enabled=mp_enabled, **le_model_cnf["model"]
    ).to(device)

    if num_gpus > 1 and not no_cuda:
        logger.info(f"Multi-GPU mode: {num_gpus} GPUs")
        model = nn.DataParallel(model)
        label_encoder = nn.DataParallel(label_encoder)
    elif not no_cuda:
        logger.info("Single-GPU mode")
    else:
        logger.info("CPU mode")
    ##################################################################################

    ############################### Prepare Training #################################
    optimizer = get_optimizer(model, label_encoder, lr, decay, le_lr, le_decay)
    scheduler = None
    scaler = GradScaler() if mp_enabled else None

    if loss_name == "circle":
        criterion = CircleLoss(m=m, gamma=gamma)
    elif loss_name == "circle2":
        criterion = CircleLoss2(m=m, gamma=gamma)
    else:
        criterion = CircleLoss3(m=m, gamma=gamma, metric=metric)

    gradient_norm_queue = (
        deque([np.inf], maxlen=5) if gradient_max_norm is not None else None
    )

    model_swa_state = {}
    le_swa_state = {}
    results = {}

    if freeze_model:
        for p in model.parameters():
            p.requires_grad_(False)

    start_epoch = 0
    global_step = 0
    best, e = 0, 0
    early_stop = False

    train_losses = deque(maxlen=print_step)

    ann_index = None

    if resume and mode == "train":
        resume_ckpt_path = (
            last_ckpt_path if os.path.exists(last_ckpt_path) else ckpt_path
        )
        if os.path.exists(resume_ckpt_path):
            logger.info("Resume Training")
            start_epoch, ckpt = load_checkpoint(
                resume_ckpt_path,
                model,
                label_encoder,
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
            le_swa_state = ckpt["le_swa_state"]
            best = ckpt["best"]
            e = ckpt["e"]

            ann_index = build_ann(
                n_candidates=ann_candidates, efS=ann_candidates, metric=metric
            )
            load_ann(ann_index, ann_index_filepath, label_embedding_filepath)

        else:
            logger.warning("No checkpoint")

    if ann_index is None and mode == "train":
        ann_index = build_ann(
            get_label_embeddings(label_encoder, device=device),
            n_candidates=ann_candidates,
            efS=ann_candidates,
            metric=metric,
        )
    ##################################################################################

    ############################### Prepare Dataloader ###############################
    logger.info(f"Preparing dataloader for {model_name}")

    if model_name in TRANSFORMER_MODELS:
        tokenizer = (
            model.module.tokenize
            if isinstance(model, nn.DataParallel)
            else model.tokenize
        )

        train_sbert_dataset = SBertDataset(
            tokenizer(train_tokenized_texts[train_mask]),
            train_dataset.y[train_mask],
        )
        valid_sbert_dataset = SBertDataset(
            tokenizer(train_tokenized_texts[~train_mask]),
            train_dataset.y[~train_mask],
        )
        test_sbert_dataset = SBertDataset(
            tokenizer(test_tokenized_texts),
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
        train_subset_dataset = Subset(train_dataset, train_ids)

        train_dataloader = DataLoader(
            IDDataset(train_subset_dataset),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=Collector(
                IDDataset(train_subset_dataset),
                ann_index,
                pos_num_labels,
                neg_num_labels,
                label_pos_neg_num,
                weight_pos_sampling,
            ),
            pin_memory=False if no_cuda else True,
        )
        valid_dataloader = DataLoader(
            Subset(IDDataset(train_dataset), valid_ids),
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        test_dataloader = DataLoader(
            IDDataset(test_dataset),
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        train_dataloader2 = DataLoader(
            IDDataset(train_subset_dataset),
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        ann_index.input_embeddings = get_embeddings(model, train_dataloader2, device)
    ##################################################################################

    logger.info(f"checkpoint name: {os.path.basename(ckpt_name)}")

    ##################################### Training ###################################
    ann_build_process = None

    if mode == "train":
        try:
            for epoch in range(start_epoch, num_epochs):
                if early_stop:
                    break

                for i, (
                    (
                        batch_anchor_doc_ids,
                        batch_anchor_doc,
                        batch_pos_labels,
                        batch_neg_labels,
                    ),
                    (batch_anchor_label_ids, batch_pos_inst, batch_neg_inst),
                ) in enumerate(train_dataloader, 1):
                    if early_stop:
                        break

                    if (
                        ann_build_process is not None
                        and not ann_build_process.is_alive()
                    ):
                        load_ann(
                            ann_index,
                            ann_index_filepath,
                            label_embedding_filepath,
                            ann_build_process,
                        )
                        logger.info("Update ANN Index")
                        ann_build_process = None

                    if (
                        global_step % eval_step == eval_step // 2
                        and ann_build_process is None
                    ):
                        logger.info("Build ann index in background")
                        ann_build_process = build_ann_async(
                            ann_index_filepath,
                            label_embedding_filepath,
                            get_label_embeddings(label_encoder, device=device),
                            n_candidates=ann_candidates,
                            efS=ann_candidates,
                            n_jobs=multiprocessing.cpu_count() // 2,
                            metric=metric,
                        )

                    train_loss = train_step(
                        model,
                        label_encoder,
                        criterion,
                        ann_index,
                        batch_anchor_doc_ids,
                        batch_anchor_doc,
                        batch_pos_labels,
                        batch_neg_labels,
                        batch_anchor_label_ids,
                        batch_pos_inst,
                        batch_neg_inst,
                        scaler,
                        optimizer,
                        gradient_clip_value=gradient_max_norm,
                        gradient_norm_queue=gradient_norm_queue,
                        device=device,
                        inv_w=inv_w_tensor if enable_loss_pos_weights else None,
                    )

                    if scheduler is not None:
                        scheduler.step()

                    train_losses.append(train_loss)

                    global_step += 1

                    if global_step == swa_warmup:
                        logger.info("Initialze SWA")
                        swa_init(model, model_swa_state)
                        swa_init(label_encoder, le_swa_state)

                    val_log_msg = ""
                    if global_step % eval_step == 0 or (
                        epoch == num_epochs - 1 and i == len(train_dataloader)
                    ):
                        # if ann_build_process is not None:
                        #     logger.info("Waiting for building ann index")
                        #     ann_build_process.join()
                        #     logger.info("Update ANN Index")
                        #     load_ann(
                        #         ann_index,
                        #         ann_index_filepath,
                        #         label_embedding_filepath,
                        #         ann_build_process,
                        #     )
                        #     ann_build_process = None

                        results = get_results(
                            model,
                            valid_dataloader,
                            train_dataset.raw_y[~train_mask],
                            ann_index,
                            mlb=mlb,
                            inv_w=inv_w,
                            device=device,
                        )

                        val_log_msg = (
                            f"p@5: {results['p5']:.5f} n@5: {results['n5']:.5f} "
                        )

                        if "psp5" in results:
                            val_log_msg += f"psp@5: {results['psp5']:.5f}"

                        if best < results[early_criterion]:
                            best = results[early_criterion]
                            e = 0
                            save_checkpoint(
                                ckpt_path,
                                model=model,
                                epoch=epoch,
                                label_encoder=label_encoder,
                                optim=optimizer,
                                scaler=scaler,
                                scheduler=scheduler,
                                results=results,
                                other_states={
                                    "best": best,
                                    "train_mask": train_mask,
                                    "train_ids": train_ids,
                                    "model_swa_state": model_swa_state,
                                    "le_swa_state": le_swa_state,
                                    "global_step": global_step,
                                    "early_criterion": early_criterion,
                                    "gradient_norm_queue": gradient_norm_queue,
                                    "e": e,
                                },
                            )
                            copy_ann_index(ann_index_filepath, best_ann_index_filepath)
                            shutil.copyfile(
                                label_embedding_filepath,
                                best_label_embedding_filepath,
                            )
                        else:
                            e += 1

                        swa_step(model, model_swa_state)
                        swa_step(label_encoder, le_swa_state)
                        swap_swa_params(model, model_swa_state)
                        swap_swa_params(label_encoder, le_swa_state)

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
            if ann_build_process is not None and ann_build_process.is_alive():
                ann_build_process.kill()
                ann_build_process.join()
                ann_build_process = None

        save_checkpoint(
            last_ckpt_path,
            model=model,
            epoch=epoch,
            label_encoder=label_encoder,
            optim=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            results=results,
            other_states={
                "best": best,
                "train_mask": train_mask,
                "train_ids": train_ids,
                "model_swa_state": model_swa_state,
                "le_swa_state": le_swa_state,
                "global_step": global_step,
                "early_criterion": early_criterion,
                "gradient_norm_queue": gradient_norm_queue,
                "e": e,
            },
        )
    ##################################################################################

    ################################### Evaluation ###################################
    logger.info("Evaluation.")
    load_checkpoint(ckpt_path, model, label_encoder, set_rng_state=False)

    test_embeddings = get_embeddings(model, test_dataloader, device)

    if ann_index is None:
        ann_index = build_ann(
            n_candidates=ann_candidates, efS=ann_candidates, metric=metric
        )

    load_ann(ann_index, best_ann_index_filepath, best_label_embedding_filepath)

    _, test_neigh = ann_index.kneighbors(test_embeddings)

    results = get_precision_results(test_neigh, test_dataset.raw_y, inv_w, mlb=mlb)

    logger.info(
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
