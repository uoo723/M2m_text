#!/usr/bin/env bash

export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
export MLFLOW_EXPERIMENT_NAME=Cornet

# DATASET=EURLex-4K
# DATASET=EURLex-300
DATASET=AmazonCat13K
# DATASET=Wiki10-31K
# DATASET=AmazonCat-1000
# DATASET=Wiki10-3000

# MODEL=SBert
# MODEL=AttentionRNNEncoder2
# MODEL=AttentionRNNEncoder
# MODEL=AttentionRNN
MODEL=LaRoberta
# MODEL=LaCNN

# LE_MODEL=LabelEncoder
# LE_MODEL=LabelGINEncoder

# CKPT_ROOT_PATH="--ckpt-root-path /results/checkpoint"

args=(
    --model-cnf config/models/$MODEL.yaml
    --data-cnf config/datasets/$DATASET.yaml
    --run-script $0
    # --test-run
    $LOG_DIR
    # --num-epochs 200
    --num-epochs 10  # AmazonCat13K
    --train-batch-size 32
    --test-batch-size 64
    --ckpt-name baseline_n5
    # --ckpt-name inplace_n5
    # --ckpt-name word_n5
    # --ckpt-name test
    --early-criterion 'n5'
    # --early-criterion 'psp5'
    --reset-best
    --seed $1
    # --swa-warmup 4
    --swa-warmup 1  # AmazonCat13K
    # --eval-step 300
    # --print-step 100
    --eval-step 10000  # AmazonCat13K
    --print-step 3000  # AmazonCat13K
    # --early 20
    --early 5  # AmazonCat13K
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 0
    # --mixup-enabled
    --mixup-type 'inplace2'
    # --mixup-type 'word'
    # --mixup-warmup 20
    --mixup-warmup 2  # AmazonCat13K
    # --mixup-warmup -1
    --mixup-num 2
    # --in-place-enabled
    # --in-place-ver 2
    --in-place-target-num 4
    # --flow-mixup-enabled
    --mixup-alpha 0.2
    # --enable-loss-weight
    # --no-label-smoothing
    # --resume
    # --resume-ckpt-path ./checkpoint/inplace_n5_t2_AttentionRNN_AmazonCat13K_0/ckpt_before_mixup.pt
    # --mode 'eval'
)

python main_mixup2.py "${args[@]}"
