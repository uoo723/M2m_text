#!/usr/bin/env bash

args=(
    --model-cnf config/models/FCNet_input.yaml
    --data-cnf config/datasets/RCV1.yaml
    --epoch 20
    --lr 0.001
    --train-batch-size 200
    --test-batch-size 500
    --ckpt-name M2m_input
    --no-over
    --seed 2
    --gen
    --warm 5
    --step-size 0.1
    --attack-iter 20
    --gamma 0.90
	--eval-step 50
    --swa-warmup -1
    --early-criterion 'acc'
    --net-g ./checkpoint/baseline_input_FCNet_RCV1_2.pt
)

python main.py "${args[@]}"
