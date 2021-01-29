#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_DrugReviewSmallv2.yaml
    --data-cnf config/datasets/DrugReviewSmallv2.yaml
    --epoch 200
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 800
    --test-batch-size 1000
    --ckpt-name M2m
    --no-over
    --no-over-gen
    --gamma 0.90
    --warm 10
    --gen
    --seed 1
    --early 100
    --early-criterion 'acc'
    --swa-warmup -1
    --attack-iter 1
    --eval-step 10
    # --net-t ./checkpoint/M2m_AttentionRNN_DrugReviewSmallv2_0_last.pt
    --net-g ./checkpoint/baseline_AttentionRNN_DrugReviewSmallv2_1.pt
)

python main.py "${args[@]}"
