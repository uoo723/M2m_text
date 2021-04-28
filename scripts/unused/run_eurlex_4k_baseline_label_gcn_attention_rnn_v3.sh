#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
# export MLFLOW_EXPERIMENT_NAME=Cornet

args=(
    --model-cnf config/models/LabelGCNAttentionRNNv3_EURLex-4K.yaml
    --data-cnf config/datasets/EURLex-4K.yaml
    --no-scheduler
    --run-script $0
    --epoch 60
    --lr 1e-3
    # --eta-min 1e-5
    --train-batch-size 40
    --test-batch-size 100
    --ckpt-name top_adj_0.02
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 10
    --eval-step 100
    --early 50
    # --net-t ./checkpoint/top_adj_0.02_LabelGCNAttentionRNNv3_EURLex4K_1240_last.pt
)

python main.py "${args[@]}"
