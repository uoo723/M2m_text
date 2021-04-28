#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_DrugReviewSmall_v2.yaml
    --data-cnf config/datasets/DrugReviewSmall.yaml
    --mode 'eval'
    --test-run
    --seed 0
    --net-t ./checkpoint/M2m_v2_attack_iter_AttentionRNN_DrugReviewSmall_0.pt
)

python main.py "${args[@]}"
