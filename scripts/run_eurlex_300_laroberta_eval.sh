#!/usr/bin/env bash

args=(
    --model-cnf config/models/LaRoberta.yaml
    --data-cnf config/datasets/EURLex-300_Roberta.yaml
    --mode 'eval'
    --test-run
    --no-over
    --net-t ./checkpoint/M2m_trial3_LaRoberta_EURLex_0_last.pt
)

python main.py "${args[@]}"
