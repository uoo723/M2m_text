"""
Created on 2020/12/31
@author Sangwoo Han
"""

import os
import random
from pathlib import Path

import click
import logzero
import numpy as np
import torch
from logzero import logger
from ruamel.yaml import YAML

from m2m_text.datasets import DrugReview, _get_le
from m2m_text.networks import AttentionRNN

MODEL_CLS = {"AttentionRNN": AttentionRNN}
DATASET_CLS = {"DrugReview": DrugReview}


def set_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logzero.logfile(log_path)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


@click.command()
@click.option("--test-run", is_flag=True, default=False, help="Test run mode for debug")
@click.option("--log-dir", type=click.Path(), default="./logs", help="Log dir")
@click.option("--seed", type=click.INT, default=0, help="Seed for reproducibility")
@click.option(
    "--model-cnf", type=click.Path(exists=True), help="Model config file path"
)
@click.option("--data-cnf", type=click.Path(exists=True), help="Data config file path")
@click.option("--no-cuda", is_flag=True, default=False, help="Disable cuda")
def main(test_run, log_dir, seed, model_cnf, data_cnf, no_cuda):
    if not test_run:
        set_logger(os.path.join(log_dir, "test.log"))

    if seed is not None:
        logger.info(f"seed: {seed}")
        set_seed(seed)

    device = torch.device("cpu" if no_cuda else "cuda")
    num_gpus = torch.cuda.device_count()

    yaml = YAML(typ="safe")
    model_cnf = yaml.load(Path(model_cnf))
    data_cnf = yaml.load(Path(data_cnf))

    ################################## Prepare Dataset ###############################
    dataset_name = data_cnf["name"]

    logger.info(f"Dataset: {dataset_name}")

    kwargs = {
        "label_encoder_filename": data_cnf.get("label_encoder", "label_encoder"),
        "maxlen": data_cnf.get("maxlen", 500),
    }

    train_dataset, valid_dataset = DATASET_CLS[dataset_name].splits(
        test_size=data_cnf.get("valid_size", 200),
        **kwargs,
    )

    test_dataset = DATASET_CLS[dataset_name](train=False, **kwargs)

    le = _get_le(train_dataset.le_path)
    n_classes = len(le.classes_)

    logger.info(f"# of train dataset: {len(train_dataset):,}")
    logger.info(f"# of valid dataset: {len(valid_dataset):,}")
    logger.info(f"# of test dataset: {len(test_dataset):,}")
    logger.info(f"# of classes: {n_classes:,}")

    ##################################################@###############################
    logger.info("good")


if __name__ == "__main__":
    main()
