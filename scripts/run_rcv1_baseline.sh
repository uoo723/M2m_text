#!/usr/bin/env bash

args=(
    --model-cnf config/models/FCNet.yaml
    --data-cnf config/datasets/RCV1.yaml
    --epoch 20
    --lr 0.001
    --train-batch-size 200
    --test-batch-size 500
    --ckpt-name baseline
    --no-over
    --seed 0
    --swa-warmup -1
    --early-criterion 'acc'
	--eval-step 50
)

python main.py "${args[@]}"
