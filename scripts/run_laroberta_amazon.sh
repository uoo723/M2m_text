#!/usr/bin/env bash

export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
export MLFLOW_EXPERIMENT_NAME=Cornet


DATASET=AmazonCat-13K
MODEL=LaRoberta

args=(
    --model-cnf config/models/$MODEL.yaml
    --data-cnf config/datasets/$DATASET.yaml
    --run-script $0
    # --test-run
    --num-epochs 20
    --train-batch-size 16
    --test-batch-size 32
    --accumulation-step 2
    --lr 8e-5
    --ckpt-name baseline_n5_t2
    # --ckpt-name inplace_n5
    # --ckpt-name word_n5
    # --ckpt-name test
    --early-criterion 'n5'
    # --early-criterion 'psp5'
    --reset-best
    --seed $1
    # --swa-warmup 4
    --swa-warmup 1
    # --eval-step 300
    # --print-step 100
    --eval-step 18000
    --print-step 5000
    --early 5
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 0
    # --mixup-enabled
    --mixup-type 'inplace2'
    # --mixup-type 'word'
    --mixup-warmup 2
    # --mixup-warmup -1
    --mixup-num 2
    # --in-place-enabled
    # --in-place-ver 2
    --in-place-target-num 4
    --ckpt-epoch 0
    # --flow-mixup-enabled
    --mixup-alpha 0.4
    # --enable-loss-weight
    # --no-label-smoothing
    # --resume
    # --resume-ckpt-path ./checkpoint/inplace_n5_t2_AttentionRNN_AmazonCat13K_0/ckpt_before_mixup.pt
    # --mode 'eval'
)

python main_mixup2.py "${args[@]}"
