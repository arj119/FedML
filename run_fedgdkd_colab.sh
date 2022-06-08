#!/bin/bash

set -ex

# code checking
# pyflakes .

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

wandb login $WANDB_API_KEY
wandb off

ALGORITHM="fedgdkd"
DATASET=$1
DATASET_DIR="$SCRIPT_DIR/data/$2"
PARTITION_METHOD=$3
PARTITION_ALPHA=$4
PARTITION_SEED=$5
DATASET_SAMPLE_R=$6
COMM_ROUNDS=$7
EPOCHS=$8
EXPERIMENT_ID=${9}
EXPERIMENT_REPETITIONS=${10}
CLIENT_NUM_IN_TOTAL=${11}
CLIENT_NUM_PER_ROUND=${12}
CONFIG_FILE="$SCRIPT_DIR/${13}"
DISTILLATION_DATASET_SIZE=${14}
KD_ALPHA=${15}
KD_EPOCHS=${16}

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
--client_config_file "$CONFIG_FILE" \
--distillation_dataset_size "$DISTILLATION_DATASET_SIZE" \
--kd_alpha "$KD_ALPHA" \
--kd_epochs "$KD_EPOCHS"
