#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_EURLex-300.yaml
    --data-cnf config/datasets/EURLex-300.yaml
    # --gen
    --mixup-enabled
    --warm 50
    --epoch 100
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name Mixup_v2_halv_grad
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --eval-step 50
    --max-n-labels 8
    --gamma 0.95
    --lam 0
    --attack-iter 3
    --perturb-attack "l2"
    --step-attack "inf"
    --step-size 0.01
    --sim-threshold 0.7
    --early 50
    # --net-t ./checkpoint/M2m_AttentionRNN_EURLex_$1_before_M2m.pt
    # --net-g ./checkpoint/baseline_AttentionRNN_EURLex_$1.pt
)

python main.py "${args[@]}"
