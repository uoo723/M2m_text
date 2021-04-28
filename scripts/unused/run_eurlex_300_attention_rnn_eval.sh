#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_EURLex-300.yaml
    --data-cnf config/datasets/EURLex-300.yaml
    --mode 'eval'
    --test-run
    --no-over
    --net-t ./checkpoint/Mixup_v2_AttentionRNN_EURLex_$1_last.pt
)

python main.py "${args[@]}"
