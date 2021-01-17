#!/usr/bin/env bash

args=(
    --model-cnf config/models/ResNet32.yaml
    --data-cnf config/datasets/Cifar10.yaml
    --epoch 200
    --lr 0.1
    --train-batch-size 128
    --test-batch-size 600
    --ckpt-name M2m
    --no-over
    --seed 0
    --gen
    --swa-warmup -1
    --warm 160
    --step-size 0.1
    --attack-iter 10
    --gamma 0.99
	--eval-step 100
    --early 1000
    # --net-t ./checkpoint/M2m_ResNet32_cifar10_0_last.pt
    --net-g ./checkpoint/baseline_ResNet32_cifar10_0.pt
)

python main_image.py "${args[@]}"
