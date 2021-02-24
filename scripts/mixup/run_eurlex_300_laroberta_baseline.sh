#!/usr/bin/env bash

args=(
    --model-cnf config/models/LaRoberta.yaml
    --data-cnf config/datasets/EURLex-300_Roberta.yaml
    --epoch 100
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name baseline
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --warm 50
    --swa-warmup -1
    --early 50
    --eval-step 50
)

python main.py "${args[@]}"
