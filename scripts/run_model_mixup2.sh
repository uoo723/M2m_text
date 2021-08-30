#!/usr/bin/env bash

export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
export MLFLOW_EXPERIMENT_NAME=Cornet

DATASET=EURLex-4K
# DATASET=EURLex-300
# DATASET=AmazonCat13K
# DATASET=Wiki10-31K
# DATASET=AmazonCat-1000
# DATASET=Wiki10-3000

# MODEL=SBert
# MODEL=AttentionRNNEncoder2
# MODEL=AttentionRNNEncoder
MODEL=AttentionRNN

# LE_MODEL=LabelEncoder
# LE_MODEL=LabelGINEncoder

# CKPT_ROOT_PATH="--ckpt-root-path /results/checkpoint"

args=(
    --model-cnf config/models/$MODEL.yaml
    --data-cnf config/datasets/$DATASET.yaml
    --run-script $0
    # --test-run
    $LOG_DIR
    --num-epochs 400
    --train-batch-size 128
    --test-batch-size 256
    --ckpt-name mixup2_warmup60_flowmixup_num3
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 10
    --eval-step 150
    --early 20
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 0
    --mixup-enabled
    --mixup-warmup 60
    --mixup-num 3
    # --resume
    # --mode 'eval'
)

python main_mixup2.py "${args[@]}"
