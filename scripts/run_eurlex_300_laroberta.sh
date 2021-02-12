#!/usr/bin/env bash

args=(
    --model-cnf config/models/LaRoberta.yaml
    --data-cnf config/datasets/EURLex-300_Roberta.yaml
    --gen
    --epoch 100
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name M2m_v4
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --warm 15
    --max-n-labels 8
    --gamma 0.95
    --lam 0.5
    --attack-iter 3
    --swa-warmup -1
    --early 50
    --eval-step 50
    --perturb-attack "l2"
    --step-attack "inf"
    --step-size 0.01
    --sim-threshold 0.6
    # --net-t ./checkpoint/M2m_v4_LaRoberta_EURLex_$1_before_M2m.pt
    --net-g ./checkpoint/baseline_v4_LaRoberta_EURLex_$1.pt
)

python main.py "${args[@]}"
