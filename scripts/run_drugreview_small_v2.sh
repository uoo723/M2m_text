#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_DrugReviewSmall_v2.yaml
    --data-cnf config/datasets/DrugReviewSmall.yaml
    --epoch 200
    --lr 0.001
    --train-batch-size 800
    --test-batch-size 1000
    --ckpt-name M2m_v2
    --no-over
    --gamma 0.90
    --warm 180
    --gen
    --seed 0
    --early 100
    --early-criterion 'acc'
    --eval-step 50
    --net-g ./checkpoint/baseline_AttentionRNN_DrugReviewSmall__v2_0.pt
)

python main.py "${args[@]}"
