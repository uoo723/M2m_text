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

from .loss import classwise_loss
from .metrics import get_accuracy, get_inv_propensity, get_precision_results
from .utils.data import get_label_features, get_mlb
from .utils.mixup import MixUp, mixup
from .utils.model import copy_model_parameters, get_model_outputs, save_checkpoint
from .utils.train import (
    clip_gradient,
    make_step,
    random_perturb,
    sum_t,
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
    gradient_norm_queue,
    gradient_max_norm,
) -> float:
    model.train()

    if is_transformer:
        inputs = {
            "input_ids": data_x[0],
            "attention_mask": data_x[1],
        }
        outputs = model(**inputs, return_dict=False)[0]
    else:
        outputs = model(data_x)

    loss = criterion(outputs, data_y)

    optimizer.zero_grad()
    loss.backward()
    clip_gradient(model, gradient_norm_queue, gradient_max_norm)
    optimizer.step()

    return loss.item()


def train_mixup_step(
    model,
    is_transformer,
    criterion,
    optimizer,
    data_x,
    data_y,
    mixup_fn,
    gradient_norm_queue,
    gradient_max_norm,
    n_samples_per_class,
    gamma,
    lam,
    step_size,
    random_start=True,
    attack_iter=10,
    input_opts={},
    gen_input_opts={},
    last_input_opts={},
    perturb_attack="l2",
    perturb_eps=0.5,
    step_attack="inf",
    multi_label=False,
) -> float:
    model.train()

    if input_opts:
        orig_inputs = get_model_outputs(
            model, data_x, input_opts=input_opts, is_transformer=is_transformer
        )
        if type(orig_inputs) == tuple:
            orig_inputs, other_inputs = orig_inputs[0], orig_inputs[1:]
        else:
            other_inputs = None
    else:
        orig_inputs = data_x
        other_inputs = None

    mixed_inputs = orig_inputs.clone()
    mixed_targets = data_y.clone()

    mixed_inputs1, mixed_targets1 = mixup_fn(mixed_inputs, mixed_targets)

    mixed_outputs1 = get_model_outputs(
        model, mixed_inputs1, other_inputs, last_input_opts, is_transformer
    )

    indices = torch.randperm(mixed_inputs.shape[0])
    lamda = mixup_fn.m.sample((mixed_targets.shape[1],)).to(mixed_targets.device)

    lamda_x = lamda.unsqueeze(-1)

    mixed_inputs2 = mixup(mixed_inputs1, orig_inputs[indices], lamda_x)
    mixed_targets2 = mixup(mixed_targets1, data_y[indices], lamda)

    mixed_outputs2 = get_model_outputs(
        model, mixed_inputs2, other_inputs, last_input_opts, is_transformer
    )

    # indices = torch.randperm(mixed_inputs.shape[0])
    # lamda = mixup_fn.m.sample((mixed_targets.shape[1],)).to(mixed_targets.device)
    # lamda_x = lamda.unsqueeze(-1)

    # mixed_inputs = mixup(mixed_inputs, orig_inputs[indices], lamda_x)
    # mixed_targets = mixup(mixed_targets, data_y[indices], lamda)

    # indices = torch.randperm(mixed_inputs.shape[0])
    # lamda = mixup_fn.m.sample((mixed_targets.shape[1],)).to(mixed_targets.device)
    # lamda_x = lamda.unsqueeze(-1)

    # mixed_inputs = mixup(mixed_inputs, orig_inputs[indices], lamda_x)
    # mixed_targets = mixup(mixed_targets, data_y[indices], lamda)

    # mixed_inputs1, mixed_targets1 = mixup_fn(orig_inputs, data_y)
    # mixed_inputs2, mixed_targets2 = mixup_fn(orig_inputs, data_y)
    # mixed_inputs = mixed_inputs1 + mixed_inputs2
    # mixed_targets = (mixed_targets1 + mixed_targets2).clamp()

    # r_targets = data_y.clone()

    # if multi_label:
    #     rows, cols = r_targets.nonzero(as_tuple=True)
    #     target_probs = n_samples_per_class[cols] / torch.max(n_samples_per_class)
    #     target_index = torch.bernoulli(target_probs).nonzero(as_tuple=True)[0]
    #     r_targets = 0.0
    #     r_targets[(rows[target_index], cols[target_index])] = 1.0
    #     # targets[(rows, cols)] = target_probs

    # input_dim = len(inputs.shape) - 1
    # input_gen_size = torch.Size([-1] + [1] * input_dim)

    # if random_start:
    #     random_noise = random_perturb(inputs, perturb_attack, perturb_eps)
    #     inputs = inputs + random_noise

    # for _ in range(attack_iter):
    #     inputs = inputs.clone().detach().requires_grad_(True)
    #     outputs = get_model_outputs(
    #         model, inputs, other_inputs, gen_input_opts, is_transformer
    #     )

    #     loss = lam * classwise_loss(outputs, r_targets, multi_label)
    #     (grad,) = torch.autograd.grad(loss, [inputs])

    #     inputs = inputs - make_step(grad, step_attack, step_size, input_gen_size)

    # inputs = inputs.detach()

    # with torch.no_grad():
    #     outputs = get_model_outputs(
    #         model, inputs, other_inputs, last_input_opts, is_transformer
    #     )

    # probs = torch.sigmoid(outputs)[(rows[target_index], cols[target_index])]
    # correct_mask = probs <= gamma

    # num_targets = rows[target_index][correct_mask].unique().shape[0]

    # if num_targets > 0:
    #     pass

    mean_n_labels = (mixed_targets > 0).sum(dim=-1).float().mean()

    outputs = get_model_outputs(
        model, orig_inputs, other_inputs, last_input_opts, is_transformer
    )
    # mixed_outputs = get_model_outputs(
    #     model, mixed_inputs, other_inputs, last_input_opts, is_transformer
    # )
    # outputs = get_model_outputs(
    #     model, orig_inputs, other_inputs, last_input_opts, is_transformer
    # )

    # loss = criterion(mixed_outputs, mixed_targets) + criterion(
    #     outputs, data_y
    # )
    # loss /= 2

    # loss = criterion(mixed_outputs, mixed_targets)
    loss1 = criterion(mixed_outputs1, mixed_targets1)
    loss2 = criterion(mixed_outputs2, mixed_targets2)
    loss3 = criterion(outputs, data_y)
    loss = (loss1 + loss2 + loss3) / 3

    optimizer.zero_grad()
    loss.backward()
    clip_gradient(model, gradient_norm_queue, gradient_max_norm)
    optimizer.step()

    return loss.item(), mean_n_labels.item()


def train_gen_step(
    model,
    model_seed,
    is_transformer,
    criterion,
    optimizer,
    data_x,
    data_y,
    num_classes,
    n_samples_per_class,
    imb_type,
    device,
    beta,
    gamma,
    lam,
    step_size,
    attack_iter,
    random_start,
    gradient_norm_queue,
    gradient_max_norm,
    input_opts={},
    gen_input_opts={},
    last_input_opts={},
    perturb_attack="l2",
    perturb_eps=0.5,
    step_attack="inf",
    multi_label=False,
    adj=None,
    max_n_labels=5,
    labels_f=None,
    sim_threshold=0.7,
    mixup_fn=None,
) -> Tuple[float, int, float, float]:
    model.train()
    model_seed.eval()
    batch_size = data_y.size(0)

    if input_opts:
        with torch.no_grad():
            orig_inputs = get_model_outputs(
                model_seed, data_x, input_opts=input_opts, is_transformer=is_transformer
            )
        if type(orig_inputs) == tuple:
            orig_inputs, other_inputs = orig_inputs[0], orig_inputs[1:]
        else:
            other_inputs = None
    else:
        orig_inputs = data_x
        other_inputs = None

    inputs = orig_inputs.clone()
    targets = data_y.clone()

    if mixup_fn is not None:
        inputs, targets = mixup_fn(inputs, targets)

    p_accept = None
    gen_target_rows = None
    gen_target_cols = None
    target_index = None
    rows = None

    if multi_label and mixup_fn is None:
        # Find samples with major class
        rows, cols = data_y.nonzero(as_tuple=True)
        target_probs = n_samples_per_class[cols] / torch.max(n_samples_per_class)
        target_index = torch.bernoulli(target_probs).nonzero(as_tuple=True)[0]

        # Select minor classes
        gen_rows, gen_cols = adj[cols[target_index].cpu()].nonzero()
        gen_rows = torch.from_numpy(gen_rows).long()
        gen_cols = torch.from_numpy(gen_cols).long()
        gen_probs = 1 - n_samples_per_class[gen_cols] / torch.max(n_samples_per_class)
        gen_index = torch.bernoulli(gen_probs).nonzero(as_tuple=True)[0]

        # Append minor classes in target samples
        select_idx = []
        gen_labels_list = []
        orig_labels_list = []

        for i in range(target_index.shape[0]):
            orig_labels = data_y[rows[target_index[i]]].nonzero(as_tuple=True)[0]
            n_labels = orig_labels.shape[0]

            if n_labels >= max_n_labels or rows[target_index[i]] in select_idx:
                continue

            row_index = (gen_rows[gen_index] == i).nonzero(as_tuple=True)[0]
            candidate_labels = gen_cols[gen_index][row_index]

            similarities = np.mean(
                (
                    labels_f[orig_labels.cpu()] @ labels_f[candidate_labels.cpu()].T
                ).toarray(),
                axis=0,
            )

            sim_index = similarities >= sim_threshold
            sim_mask = (
                (orig_labels.unsqueeze(1).cpu() != candidate_labels[sim_index])
                .prod(dim=0)
                .bool()
            )
            n_candidate_labels = sim_mask.sum().item()

            n_gen_labels = min(max_n_labels - n_labels, n_candidate_labels)

            if n_gen_labels == 0:
                continue

            bs = n_samples_per_class[cols[target_index[i]]]
            gs = n_samples_per_class[candidate_labels[sim_index][sim_mask]]

            delta = F.relu(bs - gs)
            p_accept = 1 - beta ** delta

            if p_accept.sum(-1) <= 0:
                continue

            idx = torch.multinomial(p_accept, n_gen_labels)

            gen_labels = candidate_labels[sim_index][sim_mask][idx].to(device)
            new_labels = torch.cat([orig_labels, gen_labels])
            data_y[rows[target_index[i]]][new_labels] = 1.0

            select_idx.append(rows[target_index[i]])
            gen_labels_list.append(gen_labels)
            orig_labels_list.append(orig_labels)

        if select_idx:
            select_idx = torch.hstack(select_idx)
            orig_labels_rows = torch.hstack(
                [torch.tensor([i] * len(t)) for i, t in enumerate(orig_labels_list)]
            )
            orig_labels_cols = torch.hstack(orig_labels_list)
            orig_idx = (orig_labels_rows, orig_labels_cols)
        else:
            orig_idx = None

        gen_targets = data_y[select_idx]

        if orig_idx:
            with torch.no_grad():
                scores = torch.sigmoid(
                    get_model_outputs(
                        model_seed,
                        inputs,
                        other_inputs,
                        last_input_opts,
                        is_transformer,
                    )
                )
            gen_targets[orig_idx] = scores[orig_idx]

        gen_target_rows = (
            torch.hstack(
                [torch.tensor([i] * len(t)) for i, t in enumerate(gen_labels_list)]
            )
            if gen_labels_list
            else []
        )

        gen_target_cols = torch.hstack(gen_labels_list) if gen_labels_list else []

    elif not multi_label and mixup_fn is None:
        if imb_type is not None:
            gen_probs = n_samples_per_class[data_y] / torch.max(n_samples_per_class)
            # Generation index
            gen_index = (1 - torch.bernoulli(gen_probs)).nonzero(as_tuple=False)
            gen_index = gen_index.squeeze()
            gen_targets = data_y[gen_index]
        else:
            gen_index = torch.arange(batch_size).squeeze()
            gen_targets = torch.randint(num_classes, (batch_size,)).to(device).long()

        bs = n_samples_per_class[data_y].repeat(gen_index.size(0), 1)
        gs = n_samples_per_class[gen_targets].unsqueeze(-1)

        delta = F.relu(bs - gs)
        p_accept = 1 - beta ** delta
        mask_valid = p_accept.sum(-1) > 0

        gen_index = gen_index[mask_valid]
        gen_targets = gen_targets[mask_valid]
        p_accept = p_accept[mask_valid]

        select_idx = torch.multinomial(p_accept, 1, replacement=True).squeeze()
        p_accept = p_accept.gather(1, select_idx.unsqueeze(-1)).squeeze()
    elif multi_label and mixup_fn is not None:  # Mixup
        gen_targets = targets
        gen_target_rows, gen_target_cols = targets.nonzero(as_tuple=True)
        select_idx = torch.arange(inputs.shape[0])

    if other_inputs:
        seed_inputs = (orig_inputs[select_idx],) + tuple(
            inputs[select_idx] for inputs in other_inputs
        )
    else:
        seed_inputs = orig_inputs[select_idx]

    seed_targets = targets[select_idx]

    gen_inputs, probs, correct_mask, max_prob, mean_prob = generation(
        model,
        model_seed,
        is_transformer,
        criterion,
        device,
        seed_inputs,
        seed_targets,
        gen_targets,
        p_accept,
        gamma,
        lam,
        step_size,
        random_start,
        attack_iter,
        gen_input_opts,
        last_input_opts,
        perturb_attack=perturb_attack,
        perturb_eps=perturb_eps,
        step_attack=step_attack,
        multi_label=multi_label,
        gen_target_rows=gen_target_rows,
        gen_target_cols=gen_target_cols,
    )

    num_gen = (
        gen_target_rows[correct_mask].unique().shape[0]
        if multi_label
        else sum_t(correct_mask)
    )

    orig_targets = targets.clone()

    if num_gen > 0:
        if multi_label:
            gen_c_idx = select_idx[gen_target_rows[correct_mask].unique()]
            inputs[gen_c_idx] = gen_inputs[gen_target_rows[correct_mask].unique()]
            # gen_targets[(gen_target_rows, gen_target_cols)][~correct_mask] = 0.0
            # targets[gen_c_idx] = gen_targets[gen_target_rows[correct_mask].unique()]
        else:
            gen_c_idx = gen_index[correct_mask]
            gen_inputs_c = gen_inputs[correct_mask]
            gen_targets_c = gen_targets[correct_mask]

            inputs[gen_c_idx] = gen_inputs_c
            targets[gen_c_idx] = gen_targets_c

    optimizer.zero_grad()

    outputs = get_model_outputs(
        model, inputs, other_inputs, last_input_opts, is_transformer
    )

    loss1 = criterion(outputs, targets)
    loss1.backward()

    outputs = get_model_outputs(model, data_x, is_transformer=is_transformer)

    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    loss2 = criterion(outputs, orig_targets)
    loss2.backward()

    for param in model.parameters():
        if param.grad is not None:
            param.grad /= 2

    clip_gradient(model, gradient_norm_queue, gradient_max_norm)
    optimizer.step()

    return ((loss1 + loss2) / 2).item(), num_gen, max_prob, mean_prob


def generation(
    model_r,
    model_g,
    is_transformer,
    criterion,
    device,
    inputs,
    seed_targets,
    targets,
    p_accept,
    gamma,
    lam,
    step_size,
    random_start=True,
    max_iter=10,
    gen_input_opts={},
    last_input_opts={},
    perturb_attack="l2",
    perturb_eps=0.5,
    step_attack="inf",
    multi_label=False,
    gen_target_rows=None,
    gen_target_cols=None,
):
    if type(inputs) == tuple:
        inputs, other_inputs = inputs[0], inputs[1:]
    else:
        other_inputs = None

    if inputs.nelement() == 0:
        return inputs, torch.tensor([]), torch.empty(0), 0.0, 0.0

    model_g.eval()
    model_r.eval()

    input_dim = len(inputs.shape) - 1
    input_gen_size = torch.Size([-1] + [1] * input_dim)

    if random_start:
        random_noise = random_perturb(inputs, perturb_attack, perturb_eps)
        # print("before:", inputs)
        # inputs = torch.clamp(inputs + random_noise, 0, 1)
        inputs = inputs + random_noise
        # print("after:", inputs)

    for _ in range(max_iter):
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs_g = get_model_outputs(
            model_g, inputs, other_inputs, gen_input_opts, is_transformer
        )
        outputs_r = get_model_outputs(
            model_r, inputs, other_inputs, gen_input_opts, is_transformer
        )

        loss = criterion(outputs_g, targets) + lam * classwise_loss(
            outputs_r, seed_targets, multi_label
        )
        (grad,) = torch.autograd.grad(loss, [inputs])

        inputs = inputs - make_step(grad, step_attack, step_size, input_gen_size)

    inputs = inputs.detach()

    with torch.no_grad():
        outputs_g = get_model_outputs(
            model_g, inputs, other_inputs, last_input_opts, is_transformer
        )

    if multi_label:
        probs_g = torch.sigmoid(outputs_g)[(gen_target_rows, gen_target_cols)]
        select_targets = targets[(gen_target_rows, gen_target_cols)]
        correct = torch.abs(probs_g - select_targets) <= gamma
    else:
        one_hot = torch.zeros_like(outputs_g)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        probs_g = torch.softmax(outputs_g, dim=1)[one_hot.bool()]
        correct = (probs_g >= gamma) * torch.bernoulli(p_accept).bool().to(device)

    if probs_g.nelement() == 0:
        max_prob = 0.0
        mean_prob = 0.0
    else:
        max_prob = probs_g.max().item()
        mean_prob = probs_g.mean().item()

    model_r.train()

    return inputs, probs_g, correct, max_prob, mean_prob


def train(
    model,
    device,
    start_epoch,
    epochs,
    lr_init,
    optimizer,
    scheduler,
    criterion,
    swa_warmup,
    gradient_max_norm,
    train_loader,
    train_over_loader,
    valid_loader,
    num_classes,
    n_samples_per_class,
    warm,
    gen,
    ckpt_path,
    beta,
    lam,
    step_size,
    attack_iter,
    random_start,
    other_states={},  # To resume training
    gamma=None,
    imb_type="longtail",
    model_seed=None,
    step=100,
    early=50,
    early_criterion="acc",
    input_opts={},  # To be passed to train_gen_step()
    gen_input_opts={},  # Te be passed to gneration()
    last_input_opts={},  # To be passed to train_gen_step() at the last phase of generation
    is_transformer=False,
    perturb_attack="l2",
    perturb_eps=0.5,
    step_attack="inf",
    multi_label=False,
    max_n_labels=5,
    sim_threshold=0.7,
    mixup_enabled=False,
    mixup_alpha=0.4,
):
    global_step, best = 0, 0.0

    n_samples_per_class_tensor = torch.tensor(n_samples_per_class).to(device)

    swa_state = other_states.get("swa_state", {})
    e = other_states.get("early", 0)
    best = other_states.get("best", 0)

    if gradient_max_norm is not None:
        gradient_norm_queue = other_states.get(
            "gradient_norm_queue", deque([np.inf], maxlen=5)
        )
    else:
        gradient_norm_queue = None

    epoch = start_epoch
    results = {"acc": 0.0, "bal_acc": 0.0}
    num_gen_list = deque(maxlen=50)

    last_ckpt_path, ext = os.path.splitext(ckpt_path)
    last_ckpt_path += "_last" + ext

    max_prob, mean_prob = 0, 0

    mean_n_labels_list = deque(maxlen=50)

    if multi_label:
        y = sp.vstack([train_loader.dataset.y, valid_loader.dataset.y])
        inv_w = get_inv_propensity(y)
        mlb = get_mlb(train_loader.dataset.le_path)
        adj = y.T @ y
        adj.setdiag(0)
        adj.eliminate_zeros()

        labels_f = (
            get_label_features(
                sp.vstack(
                    [
                        train_loader.dataset.get_sparse_features(),
                        valid_loader.dataset.get_sparse_features(),
                    ]
                ),
                y,
            )
            if gen
            else None
        )
    else:
        inv_w = None
        mlb = None
        adj = None
        labels_f = None

    if mixup_enabled:
        if multi_label:
            mixup_fn = MixUp(mixup_alpha, num_classes, n_samples_per_class_tensor)
        else:
            mixup_fn = MixUp(mixup_alpha)
    else:
        mixup_fn = None

    for epoch in range(start_epoch, epochs):
        if epoch == swa_warmup:
            swa_init(model, swa_state)

        if epoch == warm and mixup_enabled and not gen:
            logger.info("Start mixup")
            mixup_ckpt_path, ext = os.path.splitext(ckpt_path)
            mixup_ckpt_path += "_before_Mixup" + ext
            save_checkpoint(
                mixup_ckpt_path,
                model,
                optimizer,
                results,
                epoch,
                scheduler=scheduler,
                other_states=other_states,
            )

        if epoch == warm and gen:
            if mixup_enabled:
                logger.info("Start M2m with Mixup")
            else:
                logger.info("Start M2m")
            m2m_ckpt_path, ext = os.path.splitext(ckpt_path)
            m2m_ckpt_path += "_before_M2m" + ext
            save_checkpoint(
                m2m_ckpt_path,
                model,
                optimizer,
                results,
                epoch,
                scheduler=scheduler,
                other_states=other_states,
            )

        if epoch >= warm and gen and train_over_loader is not None:
            dataloader = train_over_loader
        else:
            dataloader = train_loader

        # adjust_learning_rate(optimizer, lr_init, epoch)
        for i, (batch_x, batch_y) in enumerate(dataloader, 1):
            if isinstance(batch_x, (list, tuple)):
                batch_x = tuple(batch.to(device) for batch in batch_x)
            else:
                batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            global_step += 1
            if epoch >= warm and mixup_enabled and not gen:
                loss, mean_n_labels = train_mixup_step(
                    model,
                    is_transformer,
                    criterion,
                    optimizer,
                    batch_x,
                    batch_y,
                    mixup_fn,
                    gradient_norm_queue,
                    gradient_max_norm,
                    n_samples_per_class_tensor,
                    gamma,
                    lam,
                    step_size,
                    random_start=random_start,
                    attack_iter=attack_iter,
                    input_opts=input_opts,
                    gen_input_opts=gen_input_opts,
                    last_input_opts=last_input_opts,
                    perturb_attack=perturb_attack,
                    perturb_eps=perturb_eps,
                    step_attack=step_attack,
                    multi_label=multi_label,
                )

                mean_n_labels_list.append(mean_n_labels)

            elif epoch >= warm and gen:
                # if epoch == warm and hasattr(model, "init_linear"):
                #     model.init_linear()

                loss, num_gen, max_prob, mean_prob = train_gen_step(
                    model,
                    model_seed,
                    is_transformer,
                    criterion,
                    optimizer,
                    batch_x,
                    batch_y,
                    num_classes,
                    n_samples_per_class_tensor,
                    imb_type,
                    device,
                    beta,
                    gamma,
                    lam,
                    step_size,
                    attack_iter,
                    random_start,
                    gradient_norm_queue,
                    gradient_max_norm,
                    input_opts=input_opts,
                    gen_input_opts=gen_input_opts,
                    last_input_opts=last_input_opts,
                    perturb_attack=perturb_attack,
                    perturb_eps=perturb_eps,
                    step_attack=step_attack,
                    multi_label=multi_label,
                    adj=adj,
                    max_n_labels=max_n_labels,
                    labels_f=labels_f,
                    sim_threshold=sim_threshold,
                    mixup_fn=mixup_fn,
                )

                num_gen_list.append(num_gen)

                if np.mean(num_gen_list) < 1.0 and global_step % step == 0:
                    logger.warn(
                        f"Number of generation is so small. "
                        f"max prob: {round(max_prob, 4)} "
                        f"mean prob: {round(mean_prob, 4)}"
                    )
            else:
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

                if mixup_enabled and len(mean_n_labels_list) > 0:
                    mixup_log_msg = f" avg labels: {np.mean(mean_n_labels_list):.2f}"
                else:
                    mixup_log_msg = ""

                if multi_label:
                    labels = np.concatenate(
                        [
                            predict_step(
                                model, batch[0], device, is_transformer, multi_label
                            )[1]
                            for batch in valid_loader
                        ]
                    )
                    targets = valid_loader.dataset.raw_y
                    results = get_precision_results(labels, targets, inv_w, mlb)
                else:
                    labels, targets = zip(
                        *[
                            (
                                predict_step(
                                    model, batch[0], device, is_transformer, multi_label
                                )[1],
                                batch[1].numpy(),
                            )
                            for batch in valid_loader
                        ]
                    )

                    labels = np.concatenate(labels)
                    targets = np.concatenate(targets)

                    results = get_accuracy(labels, targets)

                other_states = {
                    "swa_state": swa_state,
                    "gradient_norm_queue": gradient_norm_queue,
                    "early": e,
                    "best": best,
                }

                if results[early_criterion] > best:
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

                        save_checkpoint(
                            last_ckpt_path,
                            model,
                            optimizer,
                            results,
                            epoch,
                            scheduler=scheduler,
                            other_states=other_states,
                        )
                        return

                if len(num_gen_list) > 0:
                    gen_log_msg = (
                        f" avg gen: {round(np.mean(num_gen_list), 2)} "
                        f"max prob: {round(max_prob, 4)} "
                        f"mean prob: {round(mean_prob, 4)}"
                    )
                else:
                    gen_log_msg = ""

                log_msg = f"{epoch} {i * train_loader.batch_size} train loss: {round(loss, 5)} "
                log_msg += f"early stop: {e} "

                if multi_label:
                    log_msg += f"p@5: {round(results['p5'], 4)} "
                    log_msg += f"psp@5: {round(results['psp5'], 4)}"
                else:
                    log_msg += f"acc: {round(results['acc'], 4)} "
                    log_msg += f"bal acc: {round(results['bal_acc'], 4)}"

                log_msg += gen_log_msg
                log_msg += mixup_log_msg

                logger.info(log_msg)

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(
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
            logits = model(data_x.to(device))

        if multi_label:
            scores, labels = torch.topk(logits, topk)
            scores = torch.sigmoid(scores)
        else:
            scores = F.softmax(logits, dim=-1)
            labels = torch.argmax(scores, dim=-1)

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
            f"\np@1: {round(results['p1'], 4)}"
            f"\np@3: {round(results['p3'], 4)}"
            f"\np@5: {round(results['p5'], 4)}"
            f"\npsp@1: {round(results['psp1'], 4)}"
            f"\npsp@3: {round(results['psp3'], 4)}"
            f"\npsp@5: {round(results['psp5'], 4)}"
        )
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
