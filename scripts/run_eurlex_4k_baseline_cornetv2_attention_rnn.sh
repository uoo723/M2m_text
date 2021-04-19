#!/usr/bin/env bash

args=(
    --model-cnf config/models/CornetAttentionRNNv2_EURLex-4K.yaml
    --data-cnf config/datasets/EURLex-4K.yaml
    --no-scheduler
    --epoch 30
    --lr 1e-3
    # --eta-min 1e-5
    --train-batch-size 40
    --test-batch-size 100
    --ckpt-name baseline
    --no-over
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 10
    --eval-step 100
    --early 50
    # --net-t ./checkpoint/two_way_CornetAttentionRNNv2_EURLex4K_512_last.pt
)

python main.py "${args[@]}"
