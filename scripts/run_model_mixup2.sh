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
    --num-epochs 200
    # --num-epochs 10  # AmazonCat13K
    # --train-batch-size 32
    # --test-batch-size 64
    --train-batch-size 16  # RoBERTa
    --test-batch-size 32  # RoBERTa
    --accumulation-step 2  # RoBERTa
    --lr 5e-5  # RoBERTa
    --ckpt-name baseline_n5_t2
    # --ckpt-name inplace_n5_t8
    # --ckpt-name word_n5
    # --ckpt-name test
    --early-criterion 'n5'
    # --early-criterion 'psp5'
    --reset-best
    --seed $1
    --swa-warmup 4
    # --swa-warmup 1  # AmazonCat13K
    # --eval-step 300
    # --print-step 100
    --eval-step 700  # RoBERTa
    --print-step 230  # RoBERTa
    # --eval-step 10000  # AmazonCat13K
    # --print-step 3000  # AmazonCat13K
    --early 20
    # --early 5  # AmazonCat13K
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 0
    # --mixup-enabled
    # --mixup-lr 1e-4
    --mixup-type 'inplace2'
    # --mixup-type "word"
    --mixup-warmup 10
    # --mixup-warmup 1  # AmazonCat13K
    # --mixup-warmup -1
    # --mixup-num 4
    --mixup-num 4
    --in-place-target-num 6
    # --flow-mixup-enabled
    # --flow-alpha 0.5
    --mixup-alpha 0.4
    # --enable-loss-weight
    # --no-label-smoothing
    # --resume
    # --resume-ckpt-path ./checkpoint/inplace_n5_t4_AttentionRNN_Wiki10_31K_0/ckpt_before_mixup.pt
    # --mode 'eval'
    # --eval-ckpt-path checkpoint/baseline_n5_AttentionRNN_Wiki10_31K_0/ckpt.last.pt
)

python main_mixup2.py "${args[@]}"
