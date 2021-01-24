"""
Created on 2021/01/07
@author Sangwoo Han
"""
from collections import deque
from typing import Optional, Union

import torch
import torch.nn as nn
from logzero import logger


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
        logger.warning(
            f"Clipping gradients with total norm {round(total_norm, 5)} "
            f"and max norm {round(max_norm, 5)}"
        )


def swa_init(model, swa_state):
    logger.info("SWA Initializing")
    swa_state["models_num"] = 1
    for n, p in model.named_parameters():
        swa_state[n] = p.data.clone().detach()


def swa_step(model, swa_state):
    if swa_state:
        swa_state["models_num"] += 1
        beta = 1.0 / swa_state["models_num"]
        with torch.no_grad():
            for n, p in model.named_parameters():
                swa_state[n].mul_(1.0 - beta).add_(p.data, alpha=beta)


def swap_swa_params(model, swa_state):
    if swa_state:
        for n, p in model.named_parameters():
            p.data, swa_state[n] = swa_state[n], p.data
