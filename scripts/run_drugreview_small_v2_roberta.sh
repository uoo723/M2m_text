#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/DrugReviewSmallv2_Roberta.yaml
    --epoch 200
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name M2m
    --no-over
    --no-over-gen
    --gamma 0.90
    --warm 10
    --gen
    --seed 3
    # --early 100
    --early-criterion 'acc'
    --swa-warmup -1
    --attack-iter 1
    # --net-t ./checkpoint/M2m_Roberta_DrugReviewSmallv2_2_before_M2m.pt
    --net-g ./checkpoint/baseline_Roberta_DrugReviewSmallv2_3.pt
)

python main.py "${args[@]}"
