#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/EURLex-300_Roberta.yaml
    --mode 'eval'
    --test-run
    --no-over
    --net-t ./checkpoint/baseline_Roberta_EURLex_0.pt
)

python main.py "${args[@]}"
