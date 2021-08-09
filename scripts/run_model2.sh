#!/usr/bin/env bash

export MLFLOW_TRACKING_URI=http://115.145.135.65:5050
export MLFLOW_EXPERIMENT_NAME=Cornet

# DATASET=EURLex-4K
# DATASET=AmazonCat13K
DATASET=Wiki10-31K
# DATASET=AmazonCat-1000
# DATASET=Wiki10-3000

# MODEL=SBert
MODEL=AttentionRNNEncoder

LE_MODEL=LabelEncoder

# CKPT_ROOT_PATH="--ckpt-root-path /results/checkpoint"

args=(
    --model-cnf config/models/$MODEL.yaml
    --le-model-cnf config/models/$LE_MODEL.yaml
    --data-cnf config/datasets/$DATASET.yaml
    --run-script $0
    # --test-run
    $LOG_DIR
    --num-epochs 400
    --lr 1e-3
    # --le-lr 1e-5
    --train-batch-size 128
    --test-batch-size 256
    --ckpt-name normalized_loss_pos_weights
    # --ckpt-name euclidean
    --early-criterion 'n5'
    --seed $1
    --swa-warmup 600
    --eval-step 150
    --early 20
    --pos-num-samples 15
    --neg-num-samples 0
    --hard-neg-num-samples 30
    --mp-enabled
    --loss-name 'circle3'
    --weight-pos-sampling
    --gradient-max-norm 5.0
    --num-workers 0
    # --freeze-model
    # --resume
    # --mode 'eval'
    --gamma 0.1
    --m 0.1
    --hard-neg-candidates 5
    --hard-neg-candidates 10
    --hard-neg-candidates 15
    --hard-neg-candidates 20
    --hard-neg-candidates 30
    --hard-neg-candidates 50
    --ann-candidates 100
    --loss-pos-weights
    --loss-pos-weights-warmup -1
    --normalize-loss-pos-weights
    # --use-graph
    # --metric 'euclidean'
)

python main2.py "${args[@]}"
