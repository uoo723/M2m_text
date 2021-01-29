#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/DrugReviewSmallv2_Roberta.yaml
    --epoch 220
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name baseline
    --no-over
    --early-criterion 'acc'
    --seed 3
    --swa-warmup -1
    # --early 70
)

python main.py "${args[@]}"
