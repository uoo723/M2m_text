#!/usr/bin/env bash

python main.py --model-cnf config/AttentionRNN.yaml \
--data-cnf config/DrugReview.yaml \
--epoch 40 \
--lr 0.01 \
--train-batch-size 400 \
--test-batch-size 200 \
--ckpt-name baseline \
--no-over \
--net-t ./checkpoint/baseline_AttentionRNN_DrugReview_0.pt \
--test-run
