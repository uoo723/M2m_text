#!/usr/bin/env bash

args=(
    --model-cnf config/models/EaseAttentionRNN_EURLex-4K.yaml
    --data-cnf config/datasets/EURLex-4K.yaml
    --no-scheduler
    --epoch 80
    --lr 1e-3
    # --eta-min 1e-5
    --train-batch-size 40
    --test-batch-size 100
    --ckpt-name non_trainable
    --no-over
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 10
    --eval-step 100
    --early 50
    --net-t ./checkpoint/non_trainable_EaseAttentionRNN_EURLex4K_512_last.pt
)

python main.py "${args[@]}"
