#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN.yaml
    --data-cnf config/datasets/DrugReview.yaml
    --mode 'eval'
    --ckpt-name M2m
    --test-run
    --seed 0
    --net-t ./checkpoint/M2m_AttentionRNN_DrugReview_0.pt
)

python main.py "${args[@]}"
