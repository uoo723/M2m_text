#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_EURLex-300.yaml
    --data-cnf config/datasets/EURLex-300.yaml
    # --gen
    --mixup-enabled
    --warm 0
    --epoch 100
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name Mixup_v2_stacked_mixup_trial3
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --eval-step 50
    --max-n-labels 8
    --gamma 0.05
    --lam 0
    --attack-iter 3
    --perturb-attack "l2"
    --step-attack "l2"
    --step-size 0.1
    --sim-threshold 0.7
    --early 50
    # --net-t ./checkpoint/Mixup_v2_tail_labels_v2_AttentionRNN_EURLex_$1_before_Mixup.pt
    # --net-g ./checkpoint/baseline_v2_AttentionRNN_EURLex_$1.pt
)

python main.py "${args[@]}"
