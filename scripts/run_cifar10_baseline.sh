#!/usr/bin/env bash

args=(
    --model-cnf config/models/ResNet32.yaml
    --data-cnf config/datasets/Cifar10.yaml
    --epoch 200
    --lr 0.1
    # --no-scheduler
    --train-batch-size 128
    --test-batch-size 600
    --ckpt-name baseline
    --no-over
    --seed 0
	# --early-criterion 'acc'
	--eval-step 50
    --early 100
)

python main_image.py "${args[@]}"
