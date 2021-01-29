#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_DrugReviewSmallv2.yaml
    --data-cnf config/datasets/DrugReviewSmallv2.yaml
    --epoch 200
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 800
    --test-batch-size 1000
    --ckpt-name baseline
    --no-over
    --early-criterion 'acc'
    --seed 1
    --swa-warmup -1
    # --early 70
    --eval-step 10
)

python main.py "${args[@]}"
