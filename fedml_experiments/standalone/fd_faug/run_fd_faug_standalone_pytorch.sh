#!/usr/bin/env bash

GPU=$1

CLIENT_NUM=$2

WORKER_NUM=$3

BATCH_SIZE=$4

DATASET=$5

DATA_PATH=$6

DISTRIBUTION=$7

ROUND=$8

EPOCH=${9}

LR=${10}

OPT=${11}

CI=${12}

KD_GAMMA=${13}

CLIENT_CONFIG_FILE=${14}

python3 ./main_fd_fAug.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--partition_method $DISTRIBUTION  \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--client_optimizer $OPT \
--lr $LR \
--ci $CI \
#--comm_round $ROUND \
#--batch_size $BATCH_SIZE \
#--epochs $EPOCH \
#--kd_gamma $KD_GAMMA \
#--client_config_file $CLIENT_CONFIG_FILE

