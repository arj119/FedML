#!/bin/bash

set -ex

# code checking
# pyflakes .

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
source "$SCRIPT_DIR/.env"

conda activate fedml

#conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip list