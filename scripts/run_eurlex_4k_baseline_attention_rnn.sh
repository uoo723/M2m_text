#!/usr/bin/env bash

export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
export MLFLOW_EXPERIMENT_NAME=Cornet

args=(
    --model-cnf config/models/AttentionRNN_EURLex-4K.yaml
    --data-cnf config/datasets/EURLex-4K.yaml
    --run-script $0
    --no-scheduler
    --epoch 30
    --lr 1e-3
    # --eta-min 1e-5
    --train-batch-size 40
    --test-batch-size 100
    --ckpt-name mlflow_test
    --no-over
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 10
    --eval-step 100
    --early 50
    # --net-t ./checkpoint/baseline_AttentionRNN_EURLex4K_512_last.pt
)

python main.py "${args[@]}"
