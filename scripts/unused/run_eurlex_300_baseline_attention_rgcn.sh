#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRGCN_EURLex-300.yaml
    --data-cnf config/datasets/EURLex-300.yaml
    --epoch 200
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 200
    --test-batch-size 500
    --ckpt-name baseline_mean_pooling
    --no-over
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --eval-step 50
)

python main.py "${args[@]}"
