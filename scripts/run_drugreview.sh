#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN.yaml
    --data-cnf config/datasets/DrugReview.yaml
    --epoch 200
    --lr 0.001
    --train-batch-size 800
    --test-batch-size 1000
    --ckpt-name M2m
    --no-over
    --no-over-gen
    --gamma 0.90
    --warm 180
    --gen
    --seed 0
    --early 100
    --eval-step 50
    --early-criterion 'acc'
    --net-g ./checkpoint/baseline_AttentionRNN_DrugReview_0.pt
)

python main.py "${args[@]}"
