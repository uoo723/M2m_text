#!/usr/bin/env bash

args=(
    --model-cnf config/models/FCNet.yaml
    --data-cnf config/datasets/RCV1.yaml
    --epoch 20
    --lr 0.001
    --train-batch-size 1000
    --test-batch-size 2000
    --ckpt-name M2m_input
    --no-over
    --seed 0
    --gen
    --warm 1
    --step-size 0.1
    --attack-iter 30
    --gamma 0.90
	--eval-step 10
    --net-g ./checkpoint/baseline_input_FCNet_RCV1_1.pt
)

python main.py "${args[@]}"
