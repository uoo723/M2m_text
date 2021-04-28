#!/usr/bin/env bash

args=(
    --model-cnf config/models/CornetAttentionRNN_EURLex-300.yaml
    --data-cnf config/datasets/EURLex-300.yaml
    --no-scheduler
    --epoch 60
    --lr 1e-3
    # --eta-min 1e-5
    --train-batch-size 40
    --test-batch-size 100
    --ckpt-name baseline_torch1.2
    --no-over
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --eval-step 50
    --test-run
    # --net-t ./checkpoint/baseline_CornetAttentionRNN_EURLex_0_last.pt
)

python main.py "${args[@]}"
