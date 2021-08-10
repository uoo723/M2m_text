"""
Created on 2021/07/06
@author Sangwoo Han

Instace Anchor new version, no cluster
"""
import multiprocessing
import os
import shutil
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click
import dgl
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger
from ruamel.yaml import YAML
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

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
from m2m_text.networks import AttentionRNNEncoder, LabelEncoder, LabelGINEncoder, SBert
from m2m_text.optimizers import DenseSparseAdamW
from m2m_text.utils.data import copy_file, get_mlb
from m2m_text.utils.model import load_checkpoint2, save_checkpoint2
from m2m_text.utils.train import (
    clip_gradient,
    get_embeddings,
    get_label_embeddings,
    log_elapsed_time,
    save_embeddings,
    set_logger,
    set_seed,
    swa_init,
    swa_step,
    swap_swa_params,
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

LE_MODEL_CLS = {"LabelEncoder": LabelEncoder, "LabelGINEncoder": LabelGINEncoder}

# not supported mixed precision
NOT_SUPPORTED_MP = {
    "LabelGINEncoder": LabelGINEncoder,
}

USE_GRAPH_MODEL = {
    "LabelGINEncoder": LabelGINEncoder,
}

TRANSFORMER_MODELS = ["SBert"]


class Collector:
    def __init__(
        self,
        dataset: IDDataset[Subset[TextDataset]],
        ann_index: HNSW,
        pos_num_labels: int = 5,
        neg_num_labels: int = 5,
        ann_candidates: int = 30,
        hard_neg_candidates: List[int] = [5, 10, 15],
        label_pos_neg_num: Tuple[int, int, int] = (0, 0, 0),
        weight_pos_sampling: bool = False,
    ) -> None:
        self.dataset = dataset
        self.y = dataset.dataset.dataset.y[dataset.dataset.indices]
        self.inverted_index = self.y.T.tocsr()

        self.ann_index = ann_index
        self.pos_num_labels = pos_num_labels
        self.neg_num_labels = neg_num_labels
        self.ann_candidates = ann_candidates
        self.hard_neg_candidates = sorted(hard_neg_candidates)
        self.label_pos_neg_num = label_pos_neg_num
        self.weight_pos_sampling = weight_pos_sampling

        for candidates in self.hard_neg_candidates:
            assert (
                self.ann_candidates >= candidates
            ), "ann_candidates must be greater than or equal to negative_candidates"

    def __call__(
        self, batch: Iterable[Tuple[torch.Tensor, ...]]
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
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
            for candidates in self.hard_neg_candidates + [self.ann_candidates]:
                if len(neg) >= self.neg_num_labels:
                    break

                shuffle_idx = np.arange(candidates)
                np.random.shuffle(shuffle_idx)
                for neigh_label_id in neigh_labels[i][:candidates][shuffle_idx]:
                    if len(neg) >= self.neg_num_labels:
                        break

                    if (
                        neigh_label_id != -1
                        and neigh_label_id not in pos
                        and neigh_label_id not in neg
                    ):
                        neg.append(neigh_label_id)

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


def get_label_encoder(
    model_name: str,
    le_model_cnf: dict,
    num_labels: int,
    emb_init: Optional[np.ndarray] = None,
    mp_enabled: bool = False,
    device: torch.device = torch.device("cpu"),
    dataset: Optional[TextDataset] = None,
) -> nn.Module:
    kwargs = {}

    if model_name in NOT_SUPPORTED_MP:
        mp_enabled = False

    if model_name in USE_GRAPH_MODEL:
        assert dataset is not None
        adj = lil_matrix(dataset.y.T @ dataset.y)
        adj.setdiag(0)
        adj = adj.tocsr()
        src, dst = adj.nonzero()

        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])

        g = dgl.graph((u, v))
        kwargs = {"graph": g}

    label_encoder = LE_MODEL_CLS[model_name](
        emb_init=emb_init,
        num_labels=num_labels,
        mp_enabled=mp_enabled,
        **le_model_cnf["model"],
        **kwargs,
    ).to(device)

    return label_encoder


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
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
    inv_w: Optional[torch.Tensor] = None,
):
    model.train()
    label_encoder.train()
    mp_enabled = scaler is not None

    with torch.cuda.amp.autocast(enabled=mp_enabled):
        pos_inv_w = inv_w[batch_pos_labels].to(device) if inv_w is not None else None

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
    model: nn.Module,
    dataloader: DataLoader,
    raw_y: np.ndarray,
    ann_index: HNSW,
    mlb: Optional[MultiLabelBinarizer] = None,
    inv_w: Optional[np.ndarray] = None,
    return_embeddings: bool = False,
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

    if return_embeddings:
        return results, test_embeddings

    return (results,)


def get_optimizer(
    model: nn.Module,
    label_encoder: nn.Module,
    lr: float,
    decay: float,
    le_lr: float,
    le_decay: float,
) -> Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    models = [model, label_encoder]
    lr_list = [lr, le_lr]
    decay_list = [decay, le_decay]

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
    "--le-decay", type=click.FLOAT, default=1e-2, help="Weight decay for label encoder"
)
@click.option(
    "--le-lr", type=click.FLOAT, default=1e-3, help="learning rate for label encoder"
)
@click.option(
    "--ann-candidates", type=click.INT, default=30, help="# of ANN candidates"
)
@click.option(
    "--hard-neg-candidates",
    multiple=True,
    type=click.INT,
    default=[5, 10, 15],
    help="# of hard neg candidates",
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
    "--use-pretrained-label-emb",
    is_flag=True,
    default=False,
    help="Use pretrained label embedding",
)
@click.option(
    "--record-embeddings",
    is_flag=True,
    default=False,
    help="Save embeddings at every evaluation step",
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
    hard_neg_candidates: List[int],
    resume: bool,
    pos_num_labels: int,
    neg_num_labels: int,
    loss_name: str,
    gradient_max_norm: float,
    m: float,
    gamma: float,
    metric: str,
    label_pos_neg_num: Tuple[int, int, int],
    weight_pos_sampling: bool,
    use_pretrained_label_emb: bool,
    record_embeddings: bool,
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
    ckpt_name = f"{prefix}{model_name}_{dataset_name}_{seed}"
    ckpt_root_path = os.path.join(ckpt_root_path, ckpt_name)
    ckpt_path = os.path.join(ckpt_root_path, "ckpt.pt")
    last_ckpt_path = os.path.join(ckpt_root_path, "ckpt.last.pt")
    log_filename = "train.log"

    ann_index_filepath = os.path.join(ckpt_root_path, "ann_index")
    best_ann_index_filepath = os.path.join(ckpt_root_path, "best_ann_index")
    label_embedding_filepath = os.path.join(ckpt_root_path, "label_embeddings.npy")
    best_label_embedding_filepath = os.path.join(
        ckpt_root_path, "best_label_embeddings.npy"
    )

    embeddings_filepath = os.path.join(ckpt_root_path, "embeddings.npz")

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

    ################################# Prepare Model ##################################
    logger.info(f"Model: {model_name}")
    logger.info(f"Label Model: {le_model_name}")

    model = get_model(model_name, model_cnf, data_cnf, mp_enabled, device)

    if use_pretrained_label_emb:
        logger.info("Get label features")
        labels_f = train_dataset.get_label_features()
    else:
        labels_f = None

    label_encoder = get_label_encoder(
        le_model_name,
        le_model_cnf,
        num_labels,
        labels_f,
        mp_enabled,
        device,
        train_dataset,
    )

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
            start_epoch, ckpt = load_checkpoint2(
                resume_ckpt_path,
                [model, label_encoder],
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
        embeddings = get_label_embeddings(label_encoder, device=device)

        ann_index = build_ann(
            embeddings=embeddings,
            n_candidates=ann_candidates,
            efS=ann_candidates,
            metric=metric,
        )
    ##################################################################################

    ############################### Prepare Dataloader ###############################
    logger.info(f"Prepare Dataloader")

    # Not contioned
    if model_name in TRANSFORMER_MODELS:
        train_texts = train_dataset.raw_data()[0]
        test_texts = test_dataset.raw_data()[0]

        tokenizer = (
            model.module.tokenize
            if isinstance(model, nn.DataParallel)
            else model.tokenize
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
                weight_pos_sampling=weight_pos_sampling,
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
        full_train_dataloader = DataLoader(
            IDDataset(train_dataset),
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
                        ret = get_results(
                            model,
                            valid_dataloader,
                            train_dataset.raw_y[~train_mask],
                            ann_index,
                            mlb=mlb,
                            inv_w=inv_w,
                            return_embeddings=record_embeddings,
                            device=device,
                        )

                        results = ret[0]

                        if record_embeddings:
                            filepath, ext = os.path.splitext(embeddings_filepath)
                            filepath = f"{filepath}_{global_step}" + ext
                            save_embeddings(
                                full_train_dataloader,
                                test_dataloader,
                                model,
                                label_encoder,
                                filepath,
                                device,
                            )

                        val_log_msg = (
                            f"p@5: {results['p5']:.5f} n@5: {results['n5']:.5f} "
                        )

                        if "psp5" in results:
                            val_log_msg += f"psp@5: {results['psp5']:.5f}"

                        if best < results[early_criterion]:
                            best = results[early_criterion]
                            e = 0
                            save_checkpoint2(
                                ckpt_path,
                                epoch,
                                [model, label_encoder],
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

        save_checkpoint2(
            last_ckpt_path,
            epoch,
            [model, label_encoder],
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
    load_checkpoint2(ckpt_path, [model, label_encoder], set_rng_state=False)

    test_embeddings = get_embeddings(model, test_dataloader, device)

    if ann_index is None:
        ann_index = build_ann(
            n_candidates=ann_candidates, efS=ann_candidates, metric=metric
        )

    load_ann(ann_index, best_ann_index_filepath, best_label_embedding_filepath)

    test_neigh = ann_index.kneighbors(test_embeddings, return_distance=False)

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
