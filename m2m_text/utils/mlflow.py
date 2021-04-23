from typing import Mapping

import mlflow


def log_config(
    data_cnf: str,
    model_cnf: str,
    run_script: str = None,
    test_run: bool = False,
):
    if test_run:
        return

    mlflow.log_artifact(data_cnf, "config")
    mlflow.log_artifact(model_cnf, "config")

    if run_script is not None:
        mlflow.log_artifact(run_script, "scripts")


def log_logfile(log_filepath: str, test_run: bool = False):
    if test_run:
        return

    mlflow.log_artifact(log_filepath, "log")


def log_tag(
    model_name: str,
    dataset_name: str,
    prefix: str,
    seed: int,
    test_run: bool = False,
):
    if test_run:
        return

    mlflow.set_tags(
        {
            "mlflow.runName": f"{prefix}{model_name}_{dataset_name}_{seed}",
            "model": model_name,
            "dataset": dataset_name,
            "prefix": prefix[:-1] if prefix else None,
            "seed": seed,
        }
    )


def log_ckpt(ckpt_path: str, test_run: bool = False):
    if test_run:
        return

    mlflow.log_artifact(ckpt_path, "checkpoint")


def log_metric(results: Mapping[str, float], step: int = None, test_run: bool = False):
    if test_run:
        return

    mlflow.log_metrics(results, step=step)
