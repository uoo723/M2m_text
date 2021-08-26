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
from typing import Any, Callable, Dict, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch.nn as nn
from logzero import logger
from ruamel.yaml import YAML
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
    mixup_fn: Callable[
        [torch.Tensor, Optional[torch.Tensor]],
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    input_opts: Dict[str, Any],
    output_opts: Dict[str, Any],
    stacked_mixup_enabled: bool = False,
    double_mixup_enabled: bool = False,
    gradient_clip_value: Optional[float] = None,
    gradient_norm_queue: Optional[deque] = None,
    device: torch.device = torch.device("cpu"),
):
    model.train()

    batch_x = to_device(batch_x, device)
    batch_y = to_device(batch_y, device)

    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        hidden_outputs = model(batch_x, **input_opts)
        hidden_outputs, others = hidden_outputs[0], hidden_outputs[1:]
        outputs = model((hidden_outputs, *others), **output_opts)[0]

        mixed_inputs, mixed_targets = mixup_fn(hidden_outputs, batch_y)
        mixed_outputs = model((mixed_inputs, *others), **output_opts)[0]

        loss1 = criterion(outputs, batch_y)
        loss2 = criterion(mixed_outputs, mixed_targets)
        loss = loss1 + loss2
        # loss = loss2

        if stacked_mixup_enabled:
            indices = torch.randperm(mixed_inputs.shape[0])
            lamda = mixup_fn.m.sample((mixed_targets.shape[1],)).to(device)
            lamda_x = lamda.unsqueeze(-1)

            mixed_inputs2 = mixup(mixed_inputs, hidden_outputs[indices], lamda_x)
            mixed_targets2 = mixup(mixed_targets, batch_y[indices], lamda)
            mixed_outputs2 = model((mixed_inputs2, *others), **output_opts)[0]

            loss3 = criterion(mixed_outputs2, mixed_targets2)
            loss = loss + loss3

        elif double_mixup_enabled:
            mixed_inputs2, mixed_targets2 = mixup_fn(hidden_outputs, batch_y)
            mixed_outputs2 = model((mixed_inputs2, *others), **output_opts)[0]

            loss3 = criterion(mixed_outputs2, mixed_targets2)
            loss = loss + loss3

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
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(raw_y)

    prediction = np.concatenate(
        [
            predict_step(model, batch[0], device=device)[1]
            for batch in tqdm(dataloader, desc="Predict", leave=False)
        ]
    )

    prediction = mlb.classes_[prediction]

    if is_test:
        mlb = MultiLabelBinarizer(sparse_output=True).fit(raw_y)
        inv_w = get_inv_propensity(mlb.transform(raw_y))
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
@click.option("--resume", is_flag=True, default=False, help="Resume training")
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
    "--stacked-mixup-enabled",
    is_flag=True,
    default=False,
    help="Enable stacked-mixup",
)
@click.option(
    "--double-mixup-enabled",
    is_flag=True,
    default=False,
    help="Enable double-mixup",
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
    resume: bool,
    gradient_max_norm: float,
    mixup_enabled: bool,
    stacked_mixup_enabled: bool,
    double_mixup_enabled: bool,
    mixup_alpha: float,
    mixup_warmup: int,
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

    if not test_run:
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

    criterion = nn.BCEWithLogitsLoss()

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
                        train_loss = train_mixup_step(
                            model,
                            criterion,
                            batch_x,
                            batch_y,
                            scaler,
                            optimizer,
                            mixup_fn,
                            input_opts,
                            output_opts,
                            stacked_mixup_enabled,
                            double_mixup_enabled,
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
                            val_log_msg += f"psp@5: {results['psp5']:.5f}"

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
        model, test_dataloader, test_dataset.raw_y, mlb, inv_w, True, device
    )

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
