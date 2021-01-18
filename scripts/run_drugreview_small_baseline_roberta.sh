#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/DrugReviewSmall_Roberta.yaml
    --epoch 200
    --lr 0.001
    --train-batch-size 200
    --test-batch-size 500
    --ckpt-name baseline
    --no-over
    --early-criterion 'acc'
    --seed 0
    --swa-warmup -1
    --net-t ./checkpoint/baseline_Roberta_DrugReviewSmall_0_last.pt
)

python main.py "${args[@]}"
