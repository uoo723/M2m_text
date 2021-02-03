#!/usr/bin/env bash

args=(
    --model-cnf config/models/LaRoberta.yaml
    --data-cnf config/datasets/EURLex-300_Roberta.yaml
    --gen
    --epoch 200
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name M2m
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed 0
    --warm 20
    --max-n-labels 5
    --gamma 0.99
    --lam 0
    --attack-iter 1
    --swa-warmup -1
    # --net-t ./checkpoint/M2m_LaRoberta_EURLex_0.pt
    --net-g ./checkpoint/baseline_LaRoberta_EURLex_0.pt
)

python main.py "${args[@]}"
