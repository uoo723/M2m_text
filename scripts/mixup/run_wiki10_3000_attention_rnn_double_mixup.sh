#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_Wiki10-3000.yaml
    --data-cnf config/datasets/Wiki10-3000.yaml
    --mixup-enabled
    --double-mixup-enabled
    --warm 50
    --epoch 150
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 300
    --test-batch-size 500
    --ckpt-name Double_Mixup
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --eval-step 50
    --early 50
)

python main.py "${args[@]}"
