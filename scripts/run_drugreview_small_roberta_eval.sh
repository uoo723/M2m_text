#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/DrugReviewSmall_Roberta.yaml
    --mode 'eval'
    --test-run
    --seed 3
    --net-t ./checkpoint/M2m_Roberta_DrugReviewSmall_4.pt
)

python main.py "${args[@]}"
