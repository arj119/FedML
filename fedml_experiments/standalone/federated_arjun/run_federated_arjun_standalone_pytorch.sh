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

KD_EPOCHS=${13}

KD_LAMBDA=${14}

REVISIT_EPOCHS=${15}

CLIENT_CONFIG_FILE=${16}

python3 ./main_federated_arjun.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--partition_method $DISTRIBUTION  \
--lr $LR \
--client_optimizer $OPT \
#--client_num_per_round $WORKER_NUM \
#--client_num_in_total $CLIENT_NUM \
#--ci $CI \
#--comm_round $ROUND \
#--batch_size $BATCH_SIZE \
#--epochs $EPOCH \
#--kd_epochs $KD_EPOCHS \
#--kd_lambda $KD_LAMBDA \
#--revisit_epochs $REVISIT_EPOCHS \
#--client_config_file $CLIENT_CONFIG_FILE
