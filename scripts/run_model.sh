#!/usr/bin/env bash

# export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
# export MLFLOW_EXPERIMENT_NAME=Cornet

DATASET=EURLex-4K
# DATASET=AmazonCat13K
# DATASET=Wiki1-31K
# DATASET=AmazonCat-1000
# DATASET=Wiki10-3000

MODEL=AttentionRNN
# MODEL=CornetAttentionRNNv2
# MODEL=LabelGCNAttentionRNN
# MODEL=LabelGCNAttentionRNNv2
# MODEL=LabelGCNAttentionRNNv3
# MODEL=LabelGCNAttentionRNNv4

args=(
    --model-cnf config/models/$MODEL.yaml
    --data-cnf config/datasets/$DATASET.yaml
    --run-script $0
    # --test-run
    --no-scheduler
    --epoch 60
    --lr 1e-3
    # --eta-min 1e-5
    --train-batch-size 40
    --test-batch-size 100
    --ckpt-name baseline
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 10
    --eval-step 100
    --early 50
    # --net-t ./checkpoint/label_emb_init_LabelGCNAttentionRNNv4_EURLex4K_1240_last.pt
)

python main.py "${args[@]}"
