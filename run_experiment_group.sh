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

DATASET="$1"
DATASET_DIR="$2"

random-string() {
  cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-8} | head -n 1
}

GROUP_ID="$(random-string)"

for ALGORITHM in "${@:3}"
do
  ./run_fed_experiment.sh "$ALGORITHM" "$DATASET" "$DATASET_DIR"
done

echo "$GROUP_ID"