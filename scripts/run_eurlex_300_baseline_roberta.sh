#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/EURLex-300_Roberta.yaml
    --epoch 200
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name baseline
    --no-over
    --early-criterion 'p5'
    --seed 0
    --swa-warmup -1
    --eval-step 50
)

python main.py "${args[@]}"
