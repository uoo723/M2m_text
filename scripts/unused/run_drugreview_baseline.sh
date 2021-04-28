#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN.yaml
    --data-cnf config/datasets/DrugReview.yaml
    --epoch 200
    --lr 0.001
    --train-batch-size 800
    --test-batch-size 1000
    --ckpt-name baseline
    --no-over
    --seed 0
)

python main.py "${args[@]}"
