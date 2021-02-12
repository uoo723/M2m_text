#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_EURLex-300.yaml
    --data-cnf config/datasets/EURLex-300.yaml
    # --gen
    --warm 10
    --epoch 100
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name M2m_trial3
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed 2
    --swa-warmup -1
    --eval-step 50
    --max-n-labels 8
    --gamma 0.8
    --lam 0
    --attack-iter 3
    --perturb-attack "l2"
    --step-attack "inf"
    --step-size 0.01
    --sim-threshold 0.6
    --early 60
    --net-t ./checkpoint/M2m_trial3_AttentionRNN_EURLex_2_last.pt
    --net-g ./checkpoint/baseline_AttentionRNN_EURLex_2.pt
)

python main.py "${args[@]}"
