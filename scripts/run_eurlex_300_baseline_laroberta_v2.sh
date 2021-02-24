#!/usr/bin/env bash

args=(
    --model-cnf config/models/LaRoberta_v2.yaml
    --data-cnf config/datasets/EURLex-300_Roberta.yaml
    --epoch 100
    --lr 1e-3
    # --no-scheduler
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name baseline
    --no-over
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --eval-step 50
    # --net-t checkpoint/baseline_v3_LaRoberta_EURLex_0_last.pt
)

python main.py "${args[@]}"
