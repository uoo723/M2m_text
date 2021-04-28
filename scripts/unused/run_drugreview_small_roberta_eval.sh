#!/usr/bin/env bash

args=(
    --model-cnf config/models/Roberta.yaml
    --data-cnf config/datasets/DrugReviewSmallv2_Roberta.yaml
    --mode 'eval'
    --test-run
    --net-t ./checkpoint/baseline_Roberta_DrugReviewSmallv2_3_last.pt
)

python main.py "${args[@]}"
