#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/DrugReviewSmall_Roberta.yaml
    --epoch 300
    --lr 1e-3
    --eta-min 1e-5
    # --no-scheduler
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name baseline
    --no-over
    --early-criterion 'acc'
    --seed 1
    --swa-warmup -1
    --early 70
    --net-t ./checkpoint/baseline_Roberta_DrugReviewSmall_1_last.pt
)

python main.py "${args[@]}"
