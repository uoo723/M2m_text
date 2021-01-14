#!/usr/bin/env bash

args=(
    --model-cnf config/models/FCNet_input.yaml
    --data-cnf config/datasets/RCV1.yaml
    --mode 'eval'
    --test-run
    --seed 1
    --net-t ./checkpoint/baseline_input_FCNet_RCV1_1.pt
)

python main.py "${args[@]}"
