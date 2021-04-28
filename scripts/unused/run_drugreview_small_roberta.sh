#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/DrugReviewSmall_Roberta.yaml
    --epoch 40
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name M2m
    --no-over
    --gamma 0.90
    --warm 1
    --gen
    --seed 5
    --early 100
    --early-criterion 'acc'
    --eval-step 50
    --swa-warmup -1
    --net-t ./checkpoint/M2m_Roberta_DrugReviewSmall_5.pt
    --net-g ./checkpoint/baseline_Roberta_DrugReviewSmall_5.pt
)

python main.py "${args[@]}"
