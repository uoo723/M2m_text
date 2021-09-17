#!/usr/bin/env bash

export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
export MLFLOW_EXPERIMENT_NAME=Cornet

DATASET=EURLex-4K
MODEL=LaRoberta


args=(
    --model-cnf config/models/$MODEL.yaml
    --data-cnf config/datasets/$DATASET.yaml
    --run-script $0
    --num-epochs 150
    --train-batch-size 32
    --test-batch-size 32
    # --accumulation-step 2
    # --lr 5e-5
    # --ckpt-name baseline_n5_t2
    --ckpt-name inplace_n5_t4
    # --ckpt-name word_n5
    # --ckpt-name test
    --early-criterion 'n5'
    # --early-criterion 'psp5'
    --reset-best
    --seed $1
    --swa-warmup 4
    # --eval-step 700
    # --print-step 230
    --eval-step 300
    --print-step 100
    --early 20
    --mp-enabled
    --gradient-max-norm 5.0
    --num-workers 0
    --mixup-enabled
    # --mixup-lr 1e-4
    --mixup-type 'inplace2'
    # --mixup-type "word"
    --mixup-warmup 40
    # --mixup-warmup 1  # AmazonCat13K
    # --mixup-warmup -1
    # --mixup-num 4
    --mixup-num 2
    --in-place-target-num 6
    # --flow-mixup-enabled
    # --flow-alpha 0.5
    --mixup-alpha 0.4
    # --ckpt-epoch 20
    # --ckpt-epoch 30
    # --enable-loss-weight
    # --no-label-smoothing
    --resume
    --resume-ckpt-path ./checkpoint/inplace_n5_t2_LaRoberta_EURLex4K_0/ckpt_before_mixup.pt
    # --mode 'eval'
    # --eval-ckpt-path checkpoint/baseline_n5_AttentionRNN_Wiki10_31K_0/ckpt.last.pt
)

python main_mixup2.py "${args[@]}"
