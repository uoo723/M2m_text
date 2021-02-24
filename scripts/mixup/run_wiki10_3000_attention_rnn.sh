#!/usr/bin/env bash

args=(
    --model-cnf config/models/AttentionRNN_Wiki10-3000.yaml
    --data-cnf config/datasets/Wiki10-3000.yaml
    --mixup-enabled
    --warm 0
    --epoch 100
    --lr 1e-3
    --eta-min 1e-5
    --train-batch-size 100
    --test-batch-size 500
    --ckpt-name Mixup_v2_stacked_mixup
    --no-over
    --no-over-gen
    --early-criterion 'p5'
    --seed $1
    --swa-warmup -1
    --eval-step 50
    --early 50
    # --net-t ./checkpoint/Mixup_v2_tail_labels_v2_AttentionRNN_EURLex_$1_before_Mixup.pt
    # --net-g ./checkpoint/baseline_v2_AttentionRNN_EURLex_$1.pt
)

python main.py "${args[@]}"
