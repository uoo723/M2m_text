"""
Created on 2020/12/31
@author Sangwoo Han
"""

import copy
import os
import random
from pathlib import Path

import click
import logzero
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from logzero import logger
from ruamel.yaml import YAML
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from m2m_text.datasets import (
    RCV1,
    DrugReview,
    DrugReviewSmall,
    DrugReviewSmallv2,
    EURLex,
)
from m2m_text.metrics import get_inv_propensity
from m2m_text.networks import AttentionRNN, FCNet, RobertaForSeqClassification
from m2m_text.optimizers import DenseSparseAdam
from m2m_text.train import evaluate, train
from m2m_text.utils.data import (
    get_le,
    get_mlb,
    get_n_samples_per_class,
    get_oversampled_data,
)
from m2m_text.utils.model import load_checkpoint

MODEL_CLS = {
    "AttentionRNN": AttentionRNN,
    "FCNet": FCNet,
    "Roberta": RobertaForSeqClassification,
}

TRANSFORMER_MODELS = ["Roberta"]

DATASET_CLS = {
    "DrugReview": DrugReview,
    "RCV1": RCV1,
    "DrugReviewSmall": DrugReviewSmall,
    "DrugReviewSmallv2": DrugReviewSmallv2,
    "EURLex": EURLex,
}

MULTI_LABEL_DATASETS = ["EURLex"]


def set_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logzero.logfile(log_path)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model(model_name: str, num_labels: int, model_cnf: dict):
    model_cnf = copy.deepcopy(model_cnf)

    if model_name in TRANSFORMER_MODELS:
        pretrained_model_name = model_cnf["model"].pop("pretrained")
        network = MODEL_CLS[model_name].from_pretrained(
            pretrained_model_name, num_labels=num_labels, **model_cnf["model"]
        )
    else:
        network = MODEL_CLS[model_name](num_labels=num_labels, **model_cnf["model"])

    return network


def get_optimizer(model_name: str, network: nn.Module, lr: float, decay: float):
    if model_name in TRANSFORMER_MODELS:
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = [
            {
                "params": [
                    p
                    for n, p in network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": decay,
            },
            {
                "params": [
                    p
                    for n, p in network.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = DenseSparseAdam(param_groups, lr=lr)
    else:
        optimizer = DenseSparseAdam(network.parameters(), lr=lr, weight_decay=decay)

    return optimizer


@click.command(context_settings={"show_default": True})
@click.option(
    "--mode",
    type=click.Choice(["train", "eval"]),
    default="train",
    help="train: train and eval are executed. eval: eval only",
)
@click.option("--test-run", is_flag=True, default=False, help="Test run mode for debug")
@click.option("--log-dir", type=click.Path(), default="./logs", help="Log dir")
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
@click.option("--net-t", type=click.STRING, help="Checkpoint path of training network")
@click.option(
    "--net-g", type=click.STRING, help="Checkpoint path of generation network"
)
@click.option(
    "--num-workers", type=click.INT, default=4, help="Number of workers for data loader"
)
@click.option("--no-cuda", is_flag=True, default=False, help="Disable cuda")
@click.option(
    "--train-batch-size", type=click.INT, default=128, help="Batch size for training"
)
@click.option(
    "--test-batch-size", type=click.INT, default=256, help="Batch size for test"
)
@click.option("--gen", is_flag=True, default=False, help="Enable Data generation")
@click.option("--decay", type=click.FLOAT, default=2e-4, help="Weight decay")
@click.option("--lr", type=click.FLOAT, default=0.1, help="learning rate")
@click.option(
    "--eta-min",
    type=click.FLOAT,
    default=1e-4,
    help="Minimum learning rate for cosine annealing scheduler",
)
@click.option("--no-scheduler", is_flag=True, default=False, help="Disable scheduler")
@click.option(
    "--step-size", type=click.FLOAT, default=0.1, help="Step size in generation"
)
@click.option(
    "--beta",
    type=click.FLOAT,
    default=0.999,
    help="Hyper parameter for rejection/sampling",
)
@click.option(
    "--lam",
    type=click.FLOAT,
    default=0.5,
    help="Hyper paramter for regularization of translation",
)
@click.option(
    "--gamma", type=click.FLOAT, default=0.99, help="Threshold of the generation"
)
@click.option(
    "--attack-iter",
    type=click.INT,
    default=10,
    help="Attack iteration to generation synthetic sample",
)
@click.option(
    "--warm", type=click.INT, default=160, help="Deferred stragtegy for re-balancing"
)
@click.option("--epoch", type=click.INT, default=200, help="Total number of epochs")
@click.option(
    "--swa-warmup", type=click.INT, default=10, help="Warmup for SWA. Disable: -1"
)
@click.option(
    "--gradient-max-norm",
    type=click.FLOAT,
    help="max norm for gradient clipping",
)
@click.option("--no-over", is_flag=True, default=False, help="Disable over-sampling")
@click.option(
    "--no-over-gen",
    is_flag=True,
    default=False,
    help="Disable over-sampling at generation phase",
)
@click.option(
    "--eval-step",
    type=click.INT,
    default=100,
    help="Evaluation step during training",
)
@click.option(
    "--early",
    type=click.INT,
    default=50,
    help="Early stopping step",
)
@click.option(
    "--early-criterion",
    type=click.Choice(["acc", "bal_acc", "p1", "p3", "p5", "psp1", "psp3", "psp5"]),
    default="bal_acc",
    help="Early stopping criterion",
)
@click.option(
    "--perturb-attack",
    type=click.Choice(["l2", "inf"]),
    default="l2",
    help="Attack type for random perturbation",
)
@click.option(
    "--perturb-eps",
    type=click.FLOAT,
    default=0.5,
    help="Epsilon for random perturbation",
)
@click.option(
    "--step-attack",
    type=click.Choice(["l2", "inf"]),
    default="inf",
    help="Attack type for step phase",
)
def main(
    mode,
    test_run,
    log_dir,
    seed,
    model_cnf,
    data_cnf,
    ckpt_root_path,
    ckpt_name,
    net_t,
    net_g,
    num_workers,
    no_cuda,
    train_batch_size,
    test_batch_size,
    gen,
    decay,
    lr,
    eta_min,
    no_scheduler,
    step_size,
    beta,
    lam,
    gamma,
    attack_iter,
    warm,
    epoch,
    swa_warmup,
    gradient_max_norm,
    no_over,
    no_over_gen,
    eval_step,
    early,
    early_criterion,
    perturb_attack,
    perturb_eps,
    step_attack,
):
    yaml = YAML(typ="safe")
    model_cnf = yaml.load(Path(model_cnf))
    data_cnf = yaml.load(Path(data_cnf))

    os.makedirs(ckpt_root_path, exist_ok=True)

    model_name = model_cnf["name"]
    dataset_name = data_cnf["name"]

    prefix = "" if ckpt_name is None else f"{ckpt_name}_"
    ckpt_name = f"{prefix}{model_name}_{dataset_name}_{seed}.pt"
    ckpt_path = os.path.join(ckpt_root_path, ckpt_name)

    log_filename = os.path.splitext(ckpt_name)[0] + ".log"

    if not test_run:
        set_logger(os.path.join(log_dir, log_filename))

    if seed is not None:
        logger.info(f"seed: {seed}")
        set_seed(seed)

    device = torch.device("cpu" if no_cuda else "cuda")
    num_gpus = torch.cuda.device_count()

    is_transformer = model_name in TRANSFORMER_MODELS
    multi_label = dataset_name in MULTI_LABEL_DATASETS

    ################################## Prepare Dataset ###############################
    logger.info(f"Dataset: {dataset_name}")

    train_dataset, valid_dataset = DATASET_CLS[dataset_name].splits(
        test_size=data_cnf.get("valid_size", 200), **data_cnf["dataset"]
    )

    test_dataset = DATASET_CLS[dataset_name](train=False, **data_cnf["dataset"])

    le = get_le(train_dataset.le_path)
    num_labels = len(le.classes_)

    if type(train_dataset.y) == csr_matrix:
        y = sp.vstack([train_dataset.y, valid_dataset.y])
    else:
        y = np.concatenate([train_dataset.y, valid_dataset.y])

    n_samples_per_class = get_n_samples_per_class(y)

    logger.info(f"# of train dataset: {len(train_dataset):,}")
    logger.info(f"# of valid dataset: {len(valid_dataset):,}")
    logger.info(f"# of test dataset: {len(test_dataset):,}")
    logger.info(f"# of classes: {num_labels:,}")

    if not no_over:
        train_weights = get_oversampled_data(train_dataset, n_samples_per_class)
        train_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            sampler=WeightedRandomSampler(train_weights, len(train_weights)),
            pin_memory=False if no_cuda else True,
            batch_size=train_batch_size,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=False if no_cuda else True,
            batch_size=train_batch_size,
        )

    valid_loader = DataLoader(
        valid_dataset,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False if no_cuda else True,
        batch_size=test_batch_size,
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False if no_cuda else True,
        batch_size=test_batch_size,
    )
    ##################################################################################

    ################################# Prepare Model ##################################
    logger.info(f"Model: {model_name}")

    network = load_model(model_name, num_labels, model_cnf)

    network.to(device)

    if net_g:
        network_g = load_model(model_name, num_labels, model_cnf)
        load_checkpoint(net_g, network_g, set_rng_state=False)
        network_g.to(device)
    else:
        network_g = None

    if num_gpus > 1 and not no_cuda:
        logger.info(f"Multi-GPU mode: {num_gpus} GPUs")
        network = nn.DataParallel(network)
        if network_g is not None:
            network_g = nn.DataParallel(network_g)
    elif not no_cuda:
        logger.info("Single-GPU mode")
    else:
        logger.info("CPU mode")
    ##################################################################################

    ################################### Training #####################################
    if mode == "train":
        criteron = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
        optimizer = get_optimizer(model_name, network, lr, decay)
        if not no_scheduler:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max(3, epoch // 10), eta_min=eta_min
            )
        else:
            scheduler = None

        if net_t is not None:
            logger.info("Resume training")
            start_epoch, other_states = load_checkpoint(
                net_t, network, optimizer, scheduler, return_other_states=True
            )
            start_epoch += 1
        else:
            start_epoch = 0
            other_states = {}

        if net_g:
            # if isinstance(network, nn.DataParallel):
            #     network.module.emb.emb.weight.data = network_g.module.emb.emb.weight.data
            # else:
            #     network.emb.emb.weight.data = network_g.emb.emb.weight.data

            if not no_over_gen:
                train_weights = get_oversampled_data(train_dataset, n_samples_per_class)
                train_over_loader = DataLoader(
                    train_dataset,
                    sampler=WeightedRandomSampler(train_weights, len(train_weights)),
                    num_workers=num_workers,
                    pin_memory=False if no_cuda else True,
                    batch_size=train_batch_size,
                )
            else:
                train_over_loader = None
        else:
            train_over_loader = None

        logger.info("Training")

        train(
            network,
            device,
            start_epoch,
            epoch,
            lr,
            optimizer,
            scheduler,
            criteron,
            swa_warmup,
            gradient_max_norm,
            train_loader,
            train_over_loader,
            valid_loader,
            num_labels,
            n_samples_per_class,
            warm,
            gen,
            ckpt_path,
            beta,
            lam,
            step_size,
            attack_iter,
            other_states=other_states,
            gamma=gamma,
            random_start=True,
            model_seed=network_g,
            step=eval_step,
            early=early,
            early_criterion=early_criterion,
            is_transformer=is_transformer,
            perturb_attack=perturb_attack,
            perturb_eps=perturb_eps,
            step_attack=step_attack,
            multi_label=multi_label,
            **model_cnf.get("train", {}),
        )
    ##################################################################################

    ################################### Evaluation ###################################
    if mode == "eval":
        ckpt_path = net_t

    logger.info("Evaluation")

    load_checkpoint(ckpt_path, network, set_rng_state=False)

    if multi_label:
        inv_w = get_inv_propensity(sp.vstack([train_dataset.y, valid_dataset.y]))
        mlb = get_mlb(train_dataset.le_path)
    else:
        inv_w = None
        mlb = None

    evaluate(
        network,
        test_loader,
        num_labels,
        device,
        is_transformer,
        multi_label,
        inv_w,
        mlb,
    )
    ##################################################################################


if __name__ == "__main__":
    main()
