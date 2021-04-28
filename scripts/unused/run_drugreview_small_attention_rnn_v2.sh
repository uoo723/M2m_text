#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_DrugReviewSmall_v2.yaml
    --data-cnf config/datasets/DrugReviewSmall.yaml
    --epoch 100
    --lr 1e-3
    # --eta-min 1e-5
    --no-scheduler
    --train-batch-size 800
    --test-batch-size 1000
    --ckpt-name M2m_v2
    --no-over
    --gamma 0.95
    --warm 10
    --gen
    --seed 3
    # --early 100
    --early-criterion 'acc'
    # --perturb-attack 'inf'
    # --step-attack 'l2'
    --attack-iter 1
    --eval-step 30
    --early 100
    # --lam 1.5
    # --eval-step 50
    --swa-warmup -1
    # --net-t ./checkpoint/M2m_v2_AttentionRNN_DrugReviewSmall_2_last.pt
    --net-g ./checkpoint/baseline_v2_AttentionRNN_DrugReviewSmall_3.pt
)

python main.py "${args[@]}"
