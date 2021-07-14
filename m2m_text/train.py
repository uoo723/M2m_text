"""
Created on 2021/01/07
@author Sangwoo Han
"""
import os
from collections import deque
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from logzero import logger
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from m2m_text.utils.mlflow import log_metric

from .metrics import get_accuracy, get_inv_propensity, get_precision_results
from .utils.data import get_mlb
from .utils.model import save_checkpoint
from .utils.train import (
    clip_gradient,
    swa_init,
    swa_step,
    swap_swa_params,
)


def train_step(
    model,
    is_transformer,
    criterion,
    optimizer,
    data_x,
    data_y,
    gradient_norm_queue=None,
    gradient_max_norm=None,
) -> float:
    model.train()

    if is_transformer:
        inputs = {
            "input_ids": data_x[0],
            "attention_mask": data_x[1],
        }
        outputs = model(**inputs, return_dict=False)[0]
    else:
        if isinstance(data_x, dict):
            outputs = model(inputs= data_x, return_linear=True)
        else:
            outputs = model(data_x)
    
    loss = criterion(outputs, data_y)

    optimizer.zero_grad()
    loss.backward()
    clip_gradient(model, gradient_norm_queue, gradient_max_norm)
    optimizer.step()

    return loss.item()


def train(
    model,
    device,
    test_run,
    start_epoch,
    epochs,
    optimizer,
    scheduler,
    criterion,
    swa_warmup,
    gradient_max_norm,
    train_loader,
    valid_loader,
    ckpt_path,
    data_mlb, 
    other_states={},  # To resume training
    step=100,
    early=50,
    early_criterion="n5",
    is_transformer=False,
    multi_label=False,
):
    global_step, best = 0, 0.0

    swa_state = other_states.get("swa_state", {})
    e = other_states.get("early", 0)
    best = other_states.get("best", 0)
    global_step = other_states.get("global_step", 0)

    if gradient_max_norm is not None:
        gradient_norm_queue = other_states.get(
            "gradient_norm_queue", deque([np.inf], maxlen=5)
        )
    else:
        gradient_norm_queue = None

    epoch = start_epoch

    last_ckpt_path, ext = os.path.splitext(ckpt_path)
    last_ckpt_path += "_last" + ext

    y = sp.vstack([train_loader.dataset.y, valid_loader.dataset.y])
    inv_w = get_inv_propensity(y)
    #mlb = get_mlb(train_loader.dataset.le_path)
    mlb = data_mlb

    for epoch in range(start_epoch, epochs):
        if epoch == swa_warmup: #default 10으로 되어 있음. 
            swa_init(model, swa_state)

        # adjust_learning_rate(optimizer, lr_init, epoch)
        for i, (batch_x, batch_y) in enumerate(train_loader, 1):
            if isinstance(batch_x, (list, tuple)):
                batch_x = tuple(batch.to(device) for batch in batch_x)
            elif isinstance(batch_x, dict):
                for k, v in batch_x.items():
                    batch_x[k] = v.to(device)
                    #if k == "input_ids": print(f"x data:{batch_x[k]}, y_data:{batch_y}")
            else:
                batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            global_step += 1

            loss = train_step(
                model,
                is_transformer,
                criterion,
                optimizer,
                batch_x,
                batch_y,
                gradient_norm_queue,
                gradient_max_norm,
            )

            if global_step % step == 0:
                # print(train_inputs[0].dtype)
                swa_step(model, swa_state)
                swap_swa_params(model, swa_state)

                labels = np.concatenate(
                    [
                        predict_step(
                            model, batch[0], device, is_transformer, multi_label
                        )[1]
                        for batch in valid_loader
                    ]
                )
                targets = valid_loader.dataset.raw_y
                #labels:(3090, 5), targets:(3090,) 
                #print(f"labels:{labels}, targets:{targets}")
                results = get_precision_results(labels, targets, inv_w, mlb)
                log_metric(
                    {f"val_{k}": v for k, v in results.items()},
                    global_step,
                    test_run,
                )

                other_states = {
                    "swa_state": swa_state,
                    "gradient_norm_queue": gradient_norm_queue,
                    "early": e,
                    "best": best,
                    "global_step": global_step,
                }

                if results[early_criterion] > best: #best 성능을 갱신할 때마다 best 모델에 저장. 
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        results,
                        epoch,
                        scheduler=scheduler,
                        other_states=other_states,
                    )
                    best, e = results[early_criterion], 0

                    # if epoch >= warm and gen:
                    #     copy_model_parameters(model, model_seed)
                else:
                    e += 1
                    if early is not None and e > early:
                        last_ckpt_path, ext = os.path.splitext(ckpt_path)
                        last_ckpt_path += "_last" + ext

                        save_checkpoint(  #best 성능을 갱신하지 못했을 시, last 모델에 저장 
                            last_ckpt_path,
                            model,
                            optimizer,
                            results,
                            epoch,
                            scheduler=scheduler,
                            other_states=other_states,
                        )
                        return

                swap_swa_params(model, swa_state)

                log_msg = (
                    f"{epoch} {i * train_loader.batch_size} train loss: {loss:.5f} "
                )
                log_msg += f"early stop: {e} "
                log_msg += f"p@5: {results['p5']:.4f} "
                log_msg += f"n@5: {results['n5']:.4f} "
                log_msg += f"psp@5: {results['psp5']:.4f} "
                log_msg += f"psn@5: {results['psn5']:.4f}"

                logger.info(log_msg)

        if scheduler is not None:
            scheduler.step()

        save_checkpoint( #1 epoch을 다 돌 때마다 last로 모델에 저장 
            last_ckpt_path,
            model,
            optimizer,
            results,
            epoch,
            scheduler=scheduler,
            other_states=other_states,
        )


def predict_step(
    model: nn.Module,
    data_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    is_transformer: bool = False,
    multi_label: bool = False,
    topk: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        if is_transformer:
            inputs = {
                "input_ids": data_x[0].to(device),
                "attention_mask": data_x[1].to(device),
            }
            logits = model(**inputs, return_dict=False)[0]
        else:
            if isinstance(data_x, dict):
                for k, v in data_x.items():
                    data_x[k] = v.to(device)
                    #if k == "input_ids": print(f"x data:{data_x[k]}")
                logits = model(inputs= data_x, return_linear=True)
            else:
                logits = model(data_x.to(device))

        if multi_label:
            scores, labels = torch.topk(logits, topk)
            scores = torch.sigmoid(scores)
        else:
            scores = F.softmax(logits, dim=-1)
            labels = torch.argmax(scores, dim=-1)
        #print(f"bf_ps logit:{logits}, labels:{labels}")
        return scores.cpu(), labels.cpu()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    is_transformer: bool = False,
    multi_label: bool = False,
    inv_w: Optional[np.ndarray] = None,
    mlb: Optional[MultiLabelBinarizer] = None,
    test_run: bool = False,
):
    if multi_label:
        labels = np.concatenate(
            [
                predict_step(model, batch[0], device, is_transformer, multi_label)[1]
                for batch in tqdm(dataloader, desc="Predict")
            ]
        )
        targets = dataloader.dataset.raw_y
        results = get_precision_results(labels, targets, inv_w, mlb)

        logger.info(
            f"\np@1,3,5: {results['p1']:.4f}, {results['p3']:.4f}, {results['p5']:.4f}"
            f"\nn@1,3,5: {results['n1']:.4f}, {results['n3']:.4f}, {results['n5']:.4f}"
            f"\npsp@1,3,5: {results['psp1']:.4f}, {results['psp3']:.4f}, {results['psp5']:.4f}"
            f"\npsn@1,3,5: {results['psn1']:.4f}, {results['psn3']:.4f}, {results['psn5']:.4f}"
        )

        log_metric(results, test_run=test_run)

    else:
        labels, targets = zip(
            *[
                (
                    predict_step(model, batch[0], device, is_transformer, multi_label)[
                        1
                    ],
                    batch[1].numpy(),
                )
                for batch in tqdm(dataloader, desc="Predict")
            ]
        )

        labels = np.concatenate(labels)
        targets = np.concatenate(targets)

        results = get_accuracy(labels, targets)

        logger.info(
            f"acc: {round(results['acc'], 4)} bal acc: {round(results['bal_acc'], 4)}"
        )
