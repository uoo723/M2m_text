#!/usr/bin/env bash

TOKENIZERS_PARALLELISM=false

args=(
    --model-cnf config/models/LaRoberta.yaml
    --data-cnf config/datasets/Wiki10-3000_Roberta.yaml
    --mixup-enabled
    --stacked-mixup-enabled
    --warm 10
    --epoch 150
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name Stacked_Mixup
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --early 50
    --eval-step 50
)

python main.py "${args[@]}"
