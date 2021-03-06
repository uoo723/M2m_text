#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_EURLex-300.yaml
    --data-cnf config/datasets/EURLex-300.yaml
    --no-scheduler
    --epoch 100
    --lr 1e-3
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name baseline
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --eval-step 50
    --early 50
)

python main.py "${args[@]}"
