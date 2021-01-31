#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_DrugReviewSmallv2.yaml
    --data-cnf config/datasets/DrugReviewSmallv2.yaml
    --epoch 250
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 800
    --test-batch-size 1000
    --ckpt-name M2m
    --no-over
    --no-over-gen
    --gamma 0.95
    --warm 10
    --gen
    --seed 3
    --early 100
    --early-criterion 'acc'
    --swa-warmup -1
    --attack-iter 3
    --eval-step 10
    # --net-t ./checkpoint/M2m_AttentionRNN_DrugReviewSmallv2_3_last.pt
    --net-g ./checkpoint/baseline_AttentionRNN_DrugReviewSmallv2_3.pt
)

python main.py "${args[@]}"
