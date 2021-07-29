"""
Created on 2021/01/07
@author Sangwoo Han
"""
from collections import deque
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from logzero import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def make_step(grad, attack, step_size, shape=torch.Size([-1, 1, 1, 1])):
    """
    Reference: https://github.com/alinlab/M2m/blob/master/utils.py
    """
    if attack == "l2":
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(shape)
        scaled_grad = grad / (grad_norm + 1e-10)
        step = step_size * scaled_grad
    elif attack == "inf":
        step = step_size * torch.sign(grad)
    else:
        step = step_size * grad
    return step


def random_perturb(inputs, attack, eps):
    """
    Reference: https://github.com/alinlab/M2m/blob/master/utils.py
    """
    if attack == "inf":
        r_inputs = 2 * (torch.rand_like(inputs) - 0.5) * eps
    elif attack == "l2":
        r_inputs = (torch.rand_like(inputs) - 0.5).renorm(p=2, dim=1, maxnorm=eps)
    else:
        raise ValueError(f"Invalid atack type. {attack} (inf|l2)")
    return r_inputs


def sum_t(tensor):
    return tensor.float().sum().item()


def clip_gradient(
    model: Union[nn.Module, nn.DataParallel],
    gradient_norm_queue: deque,
    gradient_clip_value: Optional[float] = None,
    verbose: bool = False,
):
    if gradient_clip_value is None:
        return

    max_norm = max(gradient_norm_queue)
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm * gradient_clip_value
    )
    gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
    if total_norm > max_norm * gradient_clip_value:
        total_norm = total_norm.item() if hasattr(total_norm, "item") else total_norm
        max_norm = max_norm.item() if hasattr(max_norm, "item") else max_norm
        if verbose:
            logger.warning(
                f"Clipping gradients with total norm {round(total_norm, 5)} "
                f"and max norm {round(max_norm, 5)}"
            )


def swa_init(model, swa_state):
    logger.info("SWA Initializing")
    if isinstance(model, nn.DataParallel):
        model = model.module

    swa_state["models_num"] = 1
    for n, p in model.named_parameters():
        swa_state[n] = p.data.clone().detach()


def swa_step(model, swa_state):
    if not swa_state:
        return

    if isinstance(model, nn.DataParallel):
        model = model.module

    swa_state["models_num"] += 1
    beta = 1.0 / swa_state["models_num"]
    with torch.no_grad():
        for n, p in model.named_parameters():
            swa_state[n].mul_(1.0 - beta).add_(p.data, alpha=beta)


def swap_swa_params(model, swa_state):
    if not swa_state:
        return

    if isinstance(model, nn.DataParallel):
        model = model.module

    for n, p in model.named_parameters():
        p.data, swa_state[n] = swa_state[n], p.data


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


def to_device(
    inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: torch.device = torch.device("cpu"),
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    return (
        {k: v.to(device) for k, v in inputs.items()}
        if type(inputs) == dict
        else inputs.to(device)
    )
