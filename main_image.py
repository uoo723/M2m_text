"""
Created on 2020/12/31
@author Sangwoo Han
"""

import os
import random
from collections import Counter
from pathlib import Path

import click
import logzero
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from logzero import logger
from ruamel.yaml import YAML
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import CIFAR10

from m2m_image.data_loader import get_imbalanced, make_longtailed_imb, get_oversampled
from m2m_image.models import resnet32
from m2m_text.datasets import RCV1, DrugReview
from m2m_text.networks import AttentionRNN, FCNet
from m2m_text.optimizers import DenseSparseAdam
from m2m_text.train import evaluate, train
from m2m_text.utils.data import (
    get_emb_init,
    get_le,
    get_n_samples_per_class,
    get_oversampled_data,
)
from m2m_text.utils.model import load_checkpoint

MODEL_CLS = {"AttentionRNN": AttentionRNN, "FCNet": FCNet, "ResNet32": resnet32}
DATASET_CLS = {"DrugReview": DrugReview, "RCV1": RCV1, "cifar10": CIFAR10}


def set_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logzero.logfile(log_path)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


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
@click.option(
    "--n-samples", type=click.INT, default=5000, help="Restrict total number of samples"
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
@click.option("--swa-warmup", type=click.INT, default=10, help="Warmup for SWA")
@click.option(
    "--gradient-max-norm",
    type=click.FLOAT,
    default=5.0,
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
    type=click.Choice(["acc", "bal_acc"]),
    default="bal_acc",
    help="Early stopping criterion",
)
def main(
    mode,
    test_run,
    log_dir,
    seed,
    model_cnf,
    data_cnf,
    n_samples,
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

    ################################## Prepare Dataset ###############################
    logger.info(f"Dataset: {dataset_name}")

    n_classes = model_cnf["model"]["num_classes"]
    n_samples_per_class_base = [int(n_samples)] * n_classes
    n_samples_per_class_base = make_longtailed_imb(n_samples, n_classes, 100)
    n_samples_per_class_base = tuple(n_samples_per_class_base)

    n_samples_per_class = n_samples_per_class_base

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_loader, valid_loader, test_loader = get_imbalanced(
        dataset_name,
        n_samples_per_class_base,
        train_batch_size,
        transform_train,
        transform_test,
    )

    _, targets = zip(*valid_loader.dataset)
    targets = np.array(targets)

    valid_loader.dataset.y = targets

    logger.info(f"# of classes: {n_classes:,}")
    ##################################################################################

    ################################# Prepare Model ##################################
    logger.info(f"Model: {model_name}")

    network = MODEL_CLS[model_name](**model_cnf["model"])

    network.to(device)

    if net_g:
        network_g = MODEL_CLS[model_name](**model_cnf["model"])
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
        criteron = nn.CrossEntropyLoss()
        optimizer = DenseSparseAdam(network.parameters(), lr=lr, weight_decay=decay)
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
                train_over_loader, _, _ = get_oversampled(
                    dataset_name,
                    n_samples_per_class,
                    train_batch_size,
                    transform_train,
                    transform_test,
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
            n_classes,
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
            **model_cnf.get("train", {}),
        )
    ##################################################################################

    ################################### Evaluation ###################################
    if mode == "eval":
        ckpt_path = net_t

    logger.info("Evaluation")

    load_checkpoint(ckpt_path, network, set_rng_state=False)

    evaluate(network, test_loader, n_classes, device)
    ##################################################################################


if __name__ == "__main__":
    main()
