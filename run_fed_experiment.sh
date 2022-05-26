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

assert_eq() {
  local expected="$1"
  local actual="$2"
  local msg

  if [ "$expected" == "$actual" ]; then
    return 0
  else
    echo "$expected != $actual"
    return 1
  fi
}

round() {
  printf "%.${2}f" "${1}"
}

# 1. MNIST standalone FedAvg
cd ./fedml_experiments/standalone/"$ALGORITHM"
sh run_"$ALGORITHM"_standalone_pytorch.sh 0 2 2 4 $DATASET $DATASET_DIR hetero 2 3 0.03 sgd 1
