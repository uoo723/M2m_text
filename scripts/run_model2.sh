#!/usr/bin/env bash

export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
export MLFLOW_EXPERIMENT_NAME=Cornet

DATASET=EURLex-4K
# DATASET=AmazonCat13K
# DATASET=Wiki1-31K
# DATASET=AmazonCat-1000
# DATASET=Wiki10-3000

MODEL=SBert
# MODEL=CornetAttentionRNNv2
# MODEL=LabelGCNAttentionRNN
# MODEL=LabelGCNAttentionRNNv2
# MODEL=LabelGCNAttentionRNNv3
# MODEL=LabelGCNAttentionRNNv4
# MODEL=EaseAttentionRNN

LE_MODEL=LabelEncoder

# CKPT_ROOT_PATH="--ckpt-root-path /results/checkpoint"

args=(
    --model-cnf config/models/$MODEL.yaml
    --le-model-cnf config/models/$LE_MODEL.yaml
    --data-cnf config/datasets/$DATASET.yaml
    --run-script $0
    # --test-run
    $LOG_DIR
    $CKPT_ROOT_PATH
    --num-epochs 100
    # --lr 1e-5
    --train-batch-size 64
    --test-batch-size 128
    --ckpt-name baseline
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 980
    --eval-step 300
    --early 30
    --pos-num-samples 5
    --neg-num-samples 0
    --hard-neg-num-samples 5
    --mp-enabled
    --loss-name 'circle2'
    # --freeze-model
    # --resume
)

python main2.py "${args[@]}"
