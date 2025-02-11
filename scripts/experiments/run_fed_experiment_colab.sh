#!/bin/bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

wandb login $WANDB_API_KEY
wandb off

ALGORITHM=$1
DATASET=$2
DATASET_DIR="$SCRIPT_DIR/data/$3"
PARTITION_METHOD=$4
PARTITION_ALPHA=$5
PARTITION_SEED=$6
DATASET_SAMPLE_R=$7
COMM_ROUNDS=$8
EPOCHS=$9
EXPERIMENT_ID=${10}
EXPERIMENT_REPETITIONS=${11}
CLIENT_NUM_IN_TOTAL=${12}
CLIENT_NUM_PER_ROUND=${13}
CONFIG_FILE="$SCRIPT_DIR/${14}"

cd ./fedml_experiments/standalone/"$ALGORITHM"

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
--client_num_in_total "$CLIENT_NUM_IN_TOTAL" \
--client_num_per_round "$CLIENT_NUM_PER_ROUND" \
--client_config_file "$CONFIG_FILE"
