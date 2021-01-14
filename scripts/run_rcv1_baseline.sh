#!/usr/bin/env bash

args=(
    --model-cnf config/models/FCNet.yaml
    --data-cnf config/datasets/RCV1.yaml
    --epoch 20
    --lr 0.001
    --train-batch-size 1000
    --test-batch-size 2000
    --ckpt-name baseline
    --no-over
    --seed 0
	--eval-step 10
)

python main.py "${args[@]}"
