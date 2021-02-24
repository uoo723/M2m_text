#!/usr/bin/env bash

args=(
    --model-cnf config/models/LaRoberta.yaml
    --data-cnf config/datasets/AmazonCat-1000_Roberta.yaml
    --mixup-enabled
    --epoch 150
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name Mixup_v5_stacked_mixup
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --warm 50
    --swa-warmup -1
    --early 50
    --eval-step 50
    # --net-t ./checkpoint/Mixup_v5_stacked_mixup_LaRoberta_EURLex_$1_last.pt
    # --net-g ./checkpoint/baseline_v5_LaRoberta_EURLex_$1.pt
)

python main.py "${args[@]}"
