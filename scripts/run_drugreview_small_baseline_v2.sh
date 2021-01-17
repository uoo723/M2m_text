#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_DrugReviewSmall_v2.yaml
    --data-cnf config/datasets/DrugReviewSmall.yaml
    --epoch 200
    --lr 0.001
    --train-batch-size 800
    --test-batch-size 1000
    --ckpt-name baseline_v2
    --no-over
    --early-criterion 'acc'
    --seed 0
)

python main.py "${args[@]}"
