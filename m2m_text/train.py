"""
Created on 2021/01/07
@author Sangwoo Han
"""
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from logzero import logger
from tqdm.auto import tqdm

from .loss import classwise_loss
from .metrics import get_accuracy
from .utils.model import save_checkpoint
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
    criterion,
    optimizer,
    data_x,
    data_y,
    gradient_norm_queue,
    gradient_max_norm,
):
    model.train()
    optimizer.zero_grad()

    outputs = model(data_x)
    loss = criterion(outputs, data_y)

    loss.backward()

    if gradient_max_norm is not None:
        clip_gradient(model, gradient_max_norm, gradient_norm_queue)

    optimizer.step()

    return loss.item()


def train_gen_step(
    model,
    model_seed,
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
):
    model.train()
    model_seed.eval()
    batch_size = data_y.size(0)

    if imb_type is not None:
        gen_probs = n_samples_per_class[data_y] / torch.max(n_samples_per_class)
        # Generation index
        gen_index = (1 - torch.bernoulli(gen_probs)).nonzero(as_tuple=False)
        gen_index = gen_index.squeeze()
        gen_targets = data_y[gen_index]
    else:
        gen_index = torch.arange(batch_size).squeeze()
        gen_targets = torch.randint(num_classes, (batch_size,)).to(device).long()

    with torch.no_grad():
        orig_inputs = model_seed(data_x, **input_opts)
        if type(orig_inputs) == tuple:
            orig_inputs, other_inputs = orig_inputs[0], orig_inputs[1:]
        else:
            other_inputs = None

        inputs = orig_inputs.clone()

    targets = data_y.clone()

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

    if other_inputs:
        seed_inputs = (orig_inputs[select_idx],) + tuple(
            inputs[select_idx] for inputs in other_inputs
        )
    else:
        seed_inputs = orig_inputs[select_idx]

    seed_targets = data_y[select_idx]

    gen_inputs, correct_mask = generation(
        model,
        model_seed,
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
    )

    num_gen = sum_t(correct_mask)
    gen_c_idx = gen_index[correct_mask]

    if num_gen > 0:
        gen_inputs_c = gen_inputs[correct_mask]
        gen_targets_c = gen_targets[correct_mask]

        inputs[gen_c_idx] = gen_inputs_c
        targets[gen_c_idx] = gen_targets_c

    if other_inputs:
        model_inputs = (inputs, *other_inputs)
    else:
        model_inputs = inputs

    outputs = model(model_inputs, **last_input_opts)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()

    if gradient_max_norm is not None:
        clip_gradient(model, gradient_max_norm, gradient_norm_queue)

    optimizer.step()

    return loss.item()


def generation(
    model_r,
    model_g,
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
):
    model_g.eval()
    model_r.eval()

    if type(inputs) == tuple:
        inputs, other_inputs = inputs[0], inputs[1:]
    else:
        other_inputs = None

    if random_start:
        random_noise = random_perturb(inputs, "l2", 0.5)
        inputs = torch.clamp(inputs + random_noise, 0, 1)

    for _ in range(max_iter):
        inputs = inputs.clone().detach().requires_grad_(True)

        if other_inputs:
            model_inputs = (inputs, *other_inputs)
        else:
            model_inputs = inputs

        outputs_g = model_g(model_inputs, **gen_input_opts)
        outputs_r = model_r(model_inputs, **gen_input_opts)

        loss = criterion(outputs_g, targets) + lam * classwise_loss(
            outputs_r, seed_targets
        )
        (grad,) = torch.autograd.grad(loss, [inputs])

        inputs = inputs - make_step(
            grad, "l2", step_size, torch.Size([-1, 1, 1])
        )
        inputs = torch.clamp(inputs, 0, 1)

    inputs = inputs.detach()

    if other_inputs:
        model_inputs = (inputs, *other_inputs)
    else:
        model_inputs = inputs

    outputs_g = model_g(model_inputs, **last_input_opts)

    one_hot = torch.zeros_like(outputs_g)
    one_hot.scatter_(1, targets.view(-1, 1), 1)
    probs_g = torch.softmax(outputs_g, dim=1)[one_hot.to(torch.bool)]

    correct = (probs_g >= gamma) * torch.bernoulli(p_accept).long().to(device)

    model_r.train()

    return inputs, correct


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
    input_opts={},  # To be passed to train_gen_step()
    gen_input_opts={},  # Te be passed to gneration()
    last_input_opts={},  # To be passed to train_gen_step() at the last phase of generation
):
    global_step, best, e = 0, 0.0, 0

    n_samples_per_class_tensor = torch.tensor(n_samples_per_class).to(device)

    swa_state = other_states.get("swa_state", {})

    if gradient_max_norm is not None:
        gradient_norm_queue = other_states.get(
            "gradient_norm_queue", deque([np.inf], maxlen=5)
        )
    else:
        gradient_norm_queue = None

    epoch = start_epoch
    results = {"acc": 0.0}

    for epoch in range(start_epoch, epochs):
        if epoch == swa_warmup:
            swa_init(model, swa_state)

        if epoch >= warm and gen and train_over_loader is not None:
            dataloader = train_over_loader
        else:
            dataloader = train_loader

        # adjust_learning_rate(optimizer, lr_init, epoch)
        for i, train_inputs in enumerate(dataloader, 1):
            train_inputs = tuple(batch.to(device) for batch in train_inputs)
            global_step += 1
            if epoch >= warm and gen:
                # print(train_inputs[0].dtype)
                loss = train_gen_step(
                    model,
                    model_seed,
                    criterion,
                    optimizer,
                    *train_inputs,
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
                )
            else:
                loss = train_step(
                    model,
                    criterion,
                    optimizer,
                    *train_inputs,
                    gradient_norm_queue,
                    gradient_max_norm,
                )
            if global_step % step == 0:
                # print(train_inputs[0].dtype)
                swa_step(model, swa_state)
                swap_swa_params(model, swa_state)

                labels = [
                    predict_step(model, batch[0].to(device))[1]
                    for batch in valid_loader
                ]
                labels = np.concatenate(labels)
                targets = valid_loader.dataset.y

                results = get_accuracy(labels, targets, num_classes)

                other_states = {
                    "swa_state": swa_state,
                    "gradient_norm_queue": gradient_norm_queue,
                }

                if results["acc"] > best:
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        results["acc"],
                        epoch,
                        scheduler=scheduler,
                        other_states=other_states,
                    )
                    best, e = results["acc"], 0
                else:
                    e += 1
                    if early is not None and e > early:
                        return

                logger.info(
                    f"{epoch} {i * train_loader.batch_size} train loss: {round(loss, 5)} "
                    f"acc: {round(results['acc'], 5)} "
                    # f"major acc: {round(results['major_acc'], 5)} "
                    # f"minor acc: {round(results['minor_acc'], 5)} "
                    # f"neutral acc: {round(results['neutral_acc'], 5)} "
                    f"early stop: {e}"
                )

        if scheduler is not None:
            scheduler.step()

    last_ckpt_path, ext = os.path.splitext(ckpt_path)
    last_ckpt_path += "_last" + ext

    save_checkpoint(
        last_ckpt_path,
        model,
        optimizer,
        results["acc"],
        epoch,
        scheduler=scheduler,
        other_states=other_states,
    )


def predict_step(model, data_x):
    model.eval()
    with torch.no_grad():
        scores = F.softmax(model(data_x), dim=-1)
        labels = torch.argmax(scores, dim=-1)
        return scores.cpu(), labels.cpu()


def evaluate(model, dataloader, targets, num_classes, device, le):
    labels = [
        predict_step(model, batch[0].to(device))[1]
        for batch in tqdm(dataloader, desc="Predict")
    ]

    labels = np.concatenate(labels)

    results = get_accuracy(le.classes_[labels], targets, num_classes)

    logger.info(
        f"acc: {round(results['acc'], 5)} "
        # f"major acc: {round(results['major_acc'], 5)} "
        # f"minor acc: {round(results['minor_acc'], 5)} "
        # f"neutral acc: {round(results['neutral_acc'], 5)} "
    )
