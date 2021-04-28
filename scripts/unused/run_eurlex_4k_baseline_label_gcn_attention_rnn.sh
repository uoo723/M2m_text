#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
# export MLFLOW_EXPERIMENT_NAME=Cornet

args=(
    --model-cnf config/models/LabelGCNAttentionRNN_EURLex-4K.yaml
    --data-cnf config/datasets/EURLex-4K.yaml
    --no-scheduler
    --run-script $0
    --epoch 60
    --lr 1e-3
    # --eta-min 1e-5
    --train-batch-size 40
    --test-batch-size 100
    --ckpt-name top_adj_20_use_b_weights
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 10
    --eval-step 100
    --early 50
    # --net-t ./checkpoint/enable_gating_LabelGCNAttentionRNN_EURLex4K_512_last.pt
)

python main.py "${args[@]}"
