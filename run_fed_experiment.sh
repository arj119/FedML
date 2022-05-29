#!/bin/bash

set -ex

# code checking
# pyflakes .

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
source "$SCRIPT_DIR/.env"

conda activate fedml

wandb login $WANDB_API_KEY
wandb off

ALGORITHM=$1
DATASET=$2
DATASET_DIR="./../../../data/$3"
PARTITION_METHOD=$4
PARTITION_ALPHA=$5
PARTITION_SEED=$6
DATASET_SAMPLE_R=$7
COMM_ROUNDS=$8
EPOCHS=$9
EXPERIMENT_ID=${10}
EXPERIMENT_REPETITIONS=${11}

# 1. MNIST standalone FedAvg
cd ./fedml_experiments/standalone/"$ALGORITHM"
#sh run_"$ALGORITHM"_standalone_pytorch.sh 0 2 2 4 $DATASET $DATASET_DIR hetero 2 3 0.03 sgd 1

python3 "./main_$ALGORITHM.py" \
--gpu 0 \
--dataset "$DATASET" \
--data_dir "$DATASET_DIR" \
--dataset_r "$DATASET_SAMPLE_R" \
--partition_method "$PARTITION_METHOD" \
--partition_alpha "$PARTITION_ALPHA" \
--partition_seed "$PARTITION_SEED" \
--comm_round "$COMM_ROUNDS" \
--experiment_id "$EXPERIMENT_ID" \
--experiment_repetitions "$EXPERIMENT_REPETITIONS" \
--epochs "$EPOCHS" \

