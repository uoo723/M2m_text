#!/usr/bin/env bash

TOKENIZERS_PARALLELISM=false

args=(
    --model-cnf config/models/LaRoberta.yaml
    --data-cnf config/datasets/Wiki10-3000_Roberta.yaml
    --mixup-enabled
    --double-mixup-enabled
    --warm 30
    --epoch 250
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 200
    --test-batch-size 500
    --ckpt-name Mixup
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --early 50
    --eval-step 50
)

python main.py "${args[@]}"
