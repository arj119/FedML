import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .data_loader import load_cifar10_data, get_dataloader, get_dataloader_test
from .datasets import CIFAR10_truncated
from ..utils.partition import get_partition_indices_train, get_partition_indices_test


def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)

    class_num = len(np.unique(y_train))

    train_user_dataidx_map, traindata_cls_counts = get_partition_indices_train(X_train, y_train,
                                                                               partition=partition_method,
                                                                               num_classes=class_num,
                                                                               num_users=client_number,
                                                                               alpha=partition_alpha)

    test_user_dataidx_map, testdata_cls_counts = get_partition_indices_test(X_test, y_test, num_classes=class_num,
                                                                            num_users=client_number,
                                                                            traindata_cls_counts=traindata_cls_counts)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(train_user_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        train_dataidxs = train_user_dataidx_map[process_id - 1]
        test_dataidxs = test_user_dataidx_map[process_id - 1]

        local_data_num = len(train_dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_test(dataset, data_dir, batch_size, batch_size,
                                                                train_dataidxs, test_dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size,
                                silo_proc_num=1):
    X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)

    class_num = len(np.unique(y_train))

    train_user_dataidx_map, traindata_cls_counts = get_partition_indices_train(X_train, y_train,
                                                                               partition=partition_method,
                                                                               num_classes=class_num,
                                                                               num_users=client_number,
                                                                               alpha=partition_alpha)

    test_user_dataidx_map, testdata_cls_counts = get_partition_indices_test(X_test, y_test, num_classes=class_num,
                                                                            num_users=client_number,
                                                                            traindata_cls_counts=traindata_cls_counts)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(train_user_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        proc_train_data = dict()
        proc_test_data = dict()
        train_dataidxs = train_user_dataidx_map[client_idx]
        test_dataidxs = test_user_dataidx_map[client_idx]

        local_data_num = len(train_dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        proc_data_idxs_train = np.array_split(train_dataidxs, silo_proc_num)
        proc_data_idxs_test = np.array_split(test_dataidxs, silo_proc_num)

        print()
        for proc_rank in range(silo_proc_num):
            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local = get_dataloader_test(dataset, data_dir, batch_size, batch_size,
                                                                    proc_data_idxs_train[proc_rank],
                                                                    proc_data_idxs_test[proc_rank])
            logging.info("client_idx = %d, proc_rank = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_idx, proc_rank, len(train_data_local), len(test_data_local)))

            proc_train_data[proc_rank] = train_data_local
            proc_test_data[proc_rank] = test_data_local

        train_data_local_dict[client_idx] = proc_train_data
        test_data_local_dict[client_idx] = proc_test_data
    return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, class_num
