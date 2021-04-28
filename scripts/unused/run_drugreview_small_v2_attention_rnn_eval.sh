#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_DrugReviewSmallv2.yaml
    --data-cnf config/datasets/DrugReviewSmallv2.yaml
    --mode 'eval'
    --test-run
    --net-t ./checkpoint/M2m_AttentionRNN_DrugReviewSmallv2_1.pt
)

python main.py "${args[@]}"
