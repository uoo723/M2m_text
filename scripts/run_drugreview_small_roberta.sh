#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/DrugReviewSmall_Roberta.yaml
    --epoch 1
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 50
    --test-batch-size 200
    --ckpt-name M2m
    --no-over
    --gamma 0.90
    --warm 0
    --gen
    --seed 1
    --early 100
    --early-criterion 'acc'
    --eval-step 50
    --swa-warmup -1
    --net-g ./checkpoint/baseline_Roberta_DrugReviewSmall_1.pt
)

python main.py "${args[@]}"
