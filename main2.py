"""
Created on 2021/07/06
@author Sangwoo Han
"""
import inspect
import multiprocessing
import os
import random
import shutil
import time
import warnings
from collections import deque
from datetime import timedelta
from functools import wraps
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import click
import logzero
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from m2m_text.anns import HNSW
from m2m_text.datasets import AmazonCat13K, EURLex4K, Wiki10, Wiki10_31K
from m2m_text.datasets.custom import IDDataset
from m2m_text.datasets.sbert import SBertDataset, collate_fn
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
from m2m_text.utils.data import get_mlb
from m2m_text.utils.model import load_checkpoint, save_checkpoint
from m2m_text.utils.train import clip_gradient, swa_init, swa_step, swap_swa_params

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


def sample_pos_neg(
    inputs: Union[np.ndarray, List[int]],
    pos_num_samples: int,
    neg_num_samples: int,
    hard_neg_num_samples: int,
    batch_y: torch.Tensor,
    ann: HNSW,
    ann_candidates: int = 100,
    hard_neg_candidates: List[int] = [5, 10, 15],
    search_by_id: bool = True,
    inv_w: Optional[np.ndarray] = None,
    weight_pos_sampling: bool = False,
    is_n_pairs: bool = False,
    g: Optional[nx.Graph] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    for candidates in hard_neg_candidates:
        assert (
            ann_candidates >= candidates
        ), "ann_candidates must be greater than or equal to negative_candidates"

    positives = []
    negatives = []

    if g is not None and is_n_pairs:
        ann_candidates = int(ann_candidates * 1.5)
        neg_num_samples = int(neg_num_samples * 1.5)
        hard_neg_num_samples = int(hard_neg_num_samples * 1.5)

    _, neigh_indices = ann.kneighbors(inputs, ann_candidates, search_by_id=search_by_id)

    for i, y in enumerate(batch_y):
        pos = y.nonzero(as_tuple=True)[0].numpy()

        p = inv_w[pos] / inv_w[pos].sum() if inv_w is not None else None

        if weight_pos_sampling:
            sim = (normalize(inputs[i : i + 1]) @ normalize(ann.embeddings[pos]).T)[0]
            p = 1 - sim
            p /= p.sum()

        positives.append(
            np.random.choice(
                pos,
                size=(pos_num_samples,),
                replace=len(pos) < pos_num_samples,
                p=p,
            )
        )

        hard_neg = []
        for candidates in hard_neg_candidates + [ann_candidates]:
            if len(hard_neg) >= hard_neg_num_samples:
                break

            shuffle_idx = np.arange(candidates)
            np.random.shuffle(shuffle_idx)
            for neigh_label_id in neigh_indices[i][:candidates][shuffle_idx]:
                if len(hard_neg) >= hard_neg_num_samples:
                    break

                if (
                    neigh_label_id != -1
                    and neigh_label_id not in pos
                    and neigh_label_id not in hard_neg
                ):
                    hard_neg.append(neigh_label_id)

        # if not is_n_pairs:
        #     assert (
        #         len(hard_neg) == hard_neg_num_samples
        #     ), "Hint: Increase ann_candidates or check if HNSW returns a lot of -1"

        random_label_id = np.arange(batch_y.shape[1])
        np.random.shuffle(random_label_id)

        neg = []
        for label_id in random_label_id:
            if len(neg) + len(hard_neg) >= neg_num_samples + hard_neg_num_samples:
                break

            if label_id not in pos and label_id not in hard_neg:
                neg.append(label_id)

        assert len(neg) + len(hard_neg) == neg_num_samples + hard_neg_num_samples

        negatives.append(
            np.concatenate(
                [np.array(hard_neg, dtype=np.int64), np.array(neg, dtype=np.int64)]
            )
        )

    return np.stack(positives), np.stack(negatives)


def get_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    **tqdm_opt,
) -> np.ndarray:
    model.eval()

    idx = []
    embedding = []

    for doc_ids, batch_x, _ in tqdm(dataloader, **tqdm_opt):
        idx.append(doc_ids.numpy())
        with torch.no_grad():
            embedding.append(
                model(to_device(batch_x, device), mp_enabled=False)[0].cpu().numpy()
            )
    idx = np.concatenate(idx)
    embedding = np.concatenate(embedding)

    return embedding[np.argsort(idx)]


def get_label_embeddings(
    label_encoder: nn.Module,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    label_embeddings = []
    label_encoder.eval()

    emb = (
        label_encoder.module.emb.emb
        if isinstance(label_encoder, nn.DataParallel)
        else label_encoder.emb.emb
    )
    label_ids = torch.arange(emb.num_embeddings)

    while label_ids.shape[0] > 0:
        with torch.no_grad():
            label_embeddings.append(
                label_encoder(label_ids[:batch_size].to(device), mp_enabled=False)
                .cpu()
                .numpy()
            )
        label_ids = label_ids[batch_size:]

    return np.concatenate(label_embeddings)


def build_ann(
    embeddings: Optional[np.ndarray] = None,
    M: int = 100,
    efC: int = 300,
    efS: int = 500,
    n_candidates: int = 500,
    metric: str = "cosine",
    n_jobs: int = -1,
    filepath: Optional[str] = None,
    embedding_filepath: Optional[str] = None,
) -> HNSW:
    index = HNSW(
        M=M, efC=efC, efS=efS, n_candidates=n_candidates, metric=metric, n_jobs=n_jobs
    )

    if embeddings is not None:
        index.fit(embeddings)

    if filepath is not None:
        assert embedding_filepath is not None

        index.save_index(filepath)
        np.save(embedding_filepath, embeddings)

    return index


def build_ann_async(
    filepath: str,
    embedding_filepath: str,
    embeddings: np.ndarray,
    M: int = 100,
    efC: int = 300,
    efS: int = 500,
    n_candidates: int = 500,
    metric: str = "cosine",
    n_jobs: int = -1,
) -> Process:
    p = Process(
        target=build_ann,
        args=(
            embeddings,
            M,
            efC,
            efS,
            n_candidates,
            metric,
            n_jobs,
            filepath,
            embedding_filepath,
        ),
    )

    p.start()

    return p


def load_ann(
    index: HNSW,
    filepath: str,
    embedding_filepath: str,
    p: Optional[Process] = None,
) -> bool:
    if p is not None:
        if p.is_alive():
            return False

        if p.exitcode != 0:
            raise Exception(f"Building process failed. exit code: {p.exitcode}")

    index.load_index(filepath)
    index.embeddings = np.load(embedding_filepath)

    return True


def copy_ann_index(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)
    shutil.copyfile(src + ".dat", dst + ".dat")


def to_device(
    inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: torch.device = torch.device("cpu"),
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    return (
        {k: v.to(device) for k, v in inputs.items()}
        if type(inputs) == dict
        else inputs.to(device)
    )


def make_batch(
    model: SBert,
    batch_ids: torch.Tensor,
    batch_x: Dict[str, torch.Tensor],
    batch_y: torch.Tensor,
    ann_index: HNSW,
    ann_candidates: int,
    hard_neg_candidates: List[int],
    pos_max_num_samples: int,
    neg_max_num_samples: int,
    hard_neg_max_num_samples: int,
    inv_w: Optional[np.ndarray] = None,
    weight_pos_sampling: bool = False,
    is_n_pairs: bool = False,
    shuffle: bool = False,
    g: Optional[nx.Graph] = None,
    input_embeddings: Optional[np.ndarray] = None,
    num_samples: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    def _make_inputs():
        anchor_inputs = (
            {k: v[anchor[:batch_size]] for k, v in batch_x.items()}
            if type(batch_x) == dict
            else batch_x[anchor[:batch_size]]
        )
        positive_labels = torch.from_numpy(positive[:batch_size])
        negative_labels = torch.from_numpy(negative[:batch_size])
        return anchor_inputs, positive_labels, negative_labels

    if g is not None:
        assert num_samples is not None and input_embeddings is not None

    batch_size = batch_y.shape[0]

    model.eval()
    with torch.no_grad():
        batch_emb = model(to_device(batch_x, device), mp_enabled=False)[0].cpu().numpy()

    pos_ids, neg_ids = sample_pos_neg(
        batch_emb,
        pos_max_num_samples,
        neg_max_num_samples,
        hard_neg_max_num_samples,
        batch_y,
        ann_index,
        ann_candidates,
        hard_neg_candidates,
        search_by_id=False,
        inv_w=inv_w,
        weight_pos_sampling=weight_pos_sampling,
        is_n_pairs=is_n_pairs,
        g=g,
    )

    if is_n_pairs:
        anchor = np.arange(batch_size)
        positive = pos_ids
        negative = neg_ids

        if g is not None:
            dst_ids = []
            for i, neg_list in enumerate(negative):
                anchor_id = batch_ids[i].cpu().item()
                dst_id = []
                for neg in neg_list:
                    path = nx.shortest_path(g, anchor_id, neg.item() + num_samples)
                    dst_id.append(path[-2])
                dst_ids.append(dst_id)
            dst_ids = np.array(dst_ids)

            sim = (
                F.normalize(torch.from_numpy(batch_emb)).unsqueeze(1)
                @ F.normalize(
                    torch.from_numpy(input_embeddings[dst_ids]), dim=-1
                ).transpose(2, 1)
            ).squeeze()

            negative = np.take_along_axis(negative, sim.argsort().numpy(), 1)[
                :, : neg_max_num_samples + hard_neg_max_num_samples
            ]

    else:
        anchor = np.array(
            [i for i in range(batch_size) for _ in range(pos_max_num_samples)]
        )
        positive = pos_ids.ravel()
        negative = neg_ids.ravel()

    assert (
        anchor.shape[0] == positive.shape[0]
    ), f"# of anchor: {anchor.shape[0]}, # of positive: {positive.shape[0]}"
    assert (
        anchor.shape[0] == negative.shape[0]
    ), f"# of anchor: {anchor.shape[0]}, # of negative: {negative.shape[0]}"

    shuffle_idx = np.arange(anchor.shape[0])

    if shuffle:
        np.random.shuffle(shuffle_idx)

    anchor = anchor[shuffle_idx]
    positive = positive[shuffle_idx]
    negative = negative[shuffle_idx]

    while len(anchor) > 0:
        yield _make_inputs()
        anchor = anchor[batch_size:]
        positive = positive[batch_size:]
        negative = negative[batch_size:]


def train_step(
    model: nn.Module,
    label_encoder: nn.Module,
    batch_anchor: Dict[str, torch.Tensor],
    batch_positive: torch.Tensor,
    batch_negative: torch.Tensor,
    criterion: nn.Module,
    scaler: Optional[GradScaler] = None,
    optim: Optional[Optimizer] = None,
    is_train: bool = True,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    inv_w: Optional[torch.Tensor] = None,
):
    model.train(is_train)
    label_encoder.train(is_train)
    mp_enabled = scaler is not None

    with torch.set_grad_enabled(is_train):
        with torch.cuda.amp.autocast(enabled=mp_enabled):
            pos_inv_w = (
                inv_w[batch_positive].to(batch_positive.device)
                if inv_w is not None
                else None
            )
            anchor_outputs = model(batch_anchor)[0]
            positive_outputs = label_encoder(batch_positive)
            negative_outputs = label_encoder(batch_negative)
            loss = (
                criterion(anchor_outputs, positive_outputs, negative_outputs, pos_inv_w)
                if len(inspect.signature(criterion.forward).parameters) == 4
                else criterion(anchor_outputs, positive_outputs, negative_outputs)
            )

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


def copy_file(src: str, dst: str) -> None:
    try:
        shutil.copyfile(src, dst)
    except shutil.SameFileError:
        pass


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
    "--hard-neg-candidates",
    multiple=True,
    type=click.INT,
    default=[5, 10, 15],
    help="# of hard neg candidates",
)
@click.option(
    "--freeze-model", is_flag=True, default=False, help="Freeze model parameters"
)
@click.option("--resume", is_flag=True, default=False, help="Resume training")
@click.option(
    "--pos-num-samples", type=click.INT, default=5, help="# of positive samples"
)
@click.option(
    "--neg-num-samples", type=click.INT, default=2, help="# of negative samples"
)
@click.option(
    "--hard-neg-num-samples",
    type=click.INT,
    default=3,
    help="# of hard negative samples",
)
@click.option(
    "--loss-name",
    type=click.Choice(["circle", "circle2", "circle3"]),
    default="circle",
    help="Loss function",
)
@click.option(
    "--weight-pos-sampling",
    is_flag=True,
    default=False,
    help="Enable weighted postive sampling",
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
    "--loss-pos-weights",
    is_flag=True,
    default=False,
    help="Enable pos weights based on inv_w",
)
@click.option(
    "--normalize-loss-pos-weights",
    is_flag=True,
    default=False,
    help="normalize loss pos weights",
)
@click.option(
    "--loss-pos-weights-warmup",
    type=click.INT,
    default=10,
    help="loss pos weights warmup",
)
@click.option(
    "--metric",
    type=click.Choice(["cosine", "euclidean"]),
    default="cosine",
    help="metric function to be used",
)
@click.option("--use-graph", is_flag=True, default=False, help="Use graph for sampling")
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
    freeze_model: bool,
    resume: bool,
    pos_num_samples: int,
    neg_num_samples: int,
    hard_neg_num_samples: int,
    loss_name: str,
    weight_pos_sampling: bool,
    gradient_max_norm: float,
    m: float,
    gamma: float,
    loss_pos_weights: bool,
    loss_pos_weights_warmup: int,
    normalize_loss_pos_weights: bool,
    metric: str,
    use_graph: bool,
):
    if loss_name != "circle3":
        assert metric == "cosine"

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

        if normalize_loss_pos_weights:
            inv_w_tensor = inv_w_tensor / inv_w_tensor.max()
            # inv_w_tensor = (inv_w_tensor - inv_w_tensor.min()) / (
            #     inv_w_tensor.max() - inv_w_tensor.min()
            # )

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

    if use_graph:
        src, dst = train_dataset.y.nonzero()
        dst += len(train_dataset)
        g = nx.Graph()
        g.add_edges_from(zip(src, dst))
    else:
        g = None
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
        id_dataset = IDDataset(train_dataset)

        train_dataloader = DataLoader(
            Subset(id_dataset, train_ids),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
        valid_dataloader = DataLoader(
            Subset(id_dataset, valid_ids),
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
        full_dataloader = DataLoader(
            id_dataset,
            batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=False if no_cuda else True,
        )
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
        logger.info("Build ANN Index")

        ann_index = build_ann(
            get_label_embeddings(label_encoder, device=device),
            n_candidates=ann_candidates,
            efS=ann_candidates,
            filepath=ann_index_filepath,
            embedding_filepath=label_embedding_filepath,
            metric=metric,
        )

    if use_graph:
        logger.info("Get input embeddings")
        input_embeddings = get_embeddings(model, full_dataloader, device)
    else:
        input_embeddings = None
    ##################################################################################

    logger.info(f"checkpoint name: {os.path.basename(ckpt_name)}")

    ##################################### Training ###################################
    ann_build_process = None

    if mode == "train":
        try:
            hard_neg_candidates = sorted(hard_neg_candidates)
            for epoch in range(start_epoch, num_epochs):
                if early_stop:
                    break

                for i, (batch_ids, batch_x, batch_y) in enumerate(train_dataloader, 1):
                    if early_stop:
                        break

                    for anchor, positive, negative in make_batch(
                        model,
                        batch_ids,
                        batch_x,
                        batch_y,
                        ann_index,
                        ann_candidates,
                        hard_neg_candidates,
                        pos_num_samples,
                        neg_num_samples,
                        hard_neg_num_samples,
                        #                 inv_w,
                        weight_pos_sampling=weight_pos_sampling,
                        is_n_pairs=loss_name != "circle",
                        g=g,
                        input_embeddings=input_embeddings,
                        num_samples=len(train_dataset),
                        device=device,
                    ):

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

                        train_loss = train_step(
                            model,
                            label_encoder,
                            to_device(anchor, device),
                            to_device(positive, device),
                            to_device(negative, device),
                            criterion,
                            scaler,
                            optimizer,
                            gradient_clip_value=gradient_max_norm,
                            gradient_norm_queue=gradient_norm_queue,
                            inv_w=inv_w_tensor
                            if loss_pos_weights
                            and global_step >= loss_pos_weights_warmup
                            else None,
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
                                        "model_swa_state": model_swa_state,
                                        "le_swa_state": le_swa_state,
                                        "global_step": global_step,
                                        "early_criterion": early_criterion,
                                        "gradient_norm_queue": gradient_norm_queue,
                                        "e": e,
                                    },
                                )
                                copy_ann_index(
                                    ann_index_filepath, best_ann_index_filepath
                                )
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

                            if ann_build_process is None:
                                ann_build_process = build_ann_async(
                                    ann_index_filepath,
                                    label_embedding_filepath,
                                    get_label_embeddings(label_encoder, device=device),
                                    n_candidates=ann_candidates,
                                    efS=ann_candidates,
                                    n_jobs=multiprocessing.cpu_count() // 2,
                                    metric=metric,
                                )

                            if use_graph:
                                input_embeddings = get_embeddings(
                                    model, full_dataloader, device
                                )
                            else:
                                input_embeddings = None

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
