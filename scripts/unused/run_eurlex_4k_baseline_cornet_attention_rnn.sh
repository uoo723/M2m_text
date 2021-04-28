#!/usr/bin/env bash

args=(
    --model-cnf config/models/CornetAttentionRNN_EURLex-4K.yaml
    --data-cnf config/datasets/EURLex-4K.yaml
    --no-scheduler
    --epoch 40
    --lr 1e-3
    # --eta-min 1e-5
    --train-batch-size 40
    --test-batch-size 100
    --ckpt-name baseline_two_blocks
    --no-over
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 10
    --eval-step 50
    --early 50
    # --net-t ./checkpoint/baseline_CornetAttentionRNN_EURLex4K_0_last.pt
)

python main.py "${args[@]}"
