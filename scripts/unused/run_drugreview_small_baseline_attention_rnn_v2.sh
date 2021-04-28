#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_DrugReviewSmall_v2.yaml
    --data-cnf config/datasets/DrugReviewSmall.yaml
    --epoch 200
    --lr 1e-3
    # --eta-min 1e-5
    --no-scheduler
    --train-batch-size 800
    --test-batch-size 1000
    --ckpt-name baseline_v2
    --no-over
    --early-criterion 'acc'
    --seed 3
    --swa-warmup -1
)

python main.py "${args[@]}"
