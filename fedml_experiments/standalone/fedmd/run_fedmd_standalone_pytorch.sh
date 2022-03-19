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

PUBLIC_DATASET_SIZE=${13}

PRETRAIN_EPOCHS_PUBLIC=${14}

PRETRAIN_EPOCHS_PRIVATE=${15}

DIGEST_EPOCHS=${16}

KD_LAMBDA=${17}

REVISIT_EPOCHS=${18}

CLIENT_CONFIG_FILE=${19}

python3 ./main_fedmd.py \
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
#--public_dataset_size $PUBLIC_DATASET_SIZE \
#--pretrain_epochs_public $PRETRAIN_EPOCHS_PUBLIC \
#--pretrain_epochs_private $PRETRAIN_EPOCHS_PRIVATE \
#--digest_epochs $DIGEST_EPOCHS \
#--kd_lambda $KD_LAMBDA \
#--revisit_epochs $REVISIT_EPOCHS \
#--client_config_file $CLIENT_CONFIG_FILE
$
