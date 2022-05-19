import logging
from collections import defaultdict
from functools import partial

import numpy as np
from torch.utils import data


def get_label_indices(y):
    label_indices = {}
    for i in np.unique(y):
        label_indices[i] = list(np.where(y == i)[0])
    return label_indices


def get_partition_indices_train(X, y, num_classes, partition, num_users, alpha, dataidx_map_file_path=None,
                                distribution_file_path=None):
    n = X.shape[0]
    net_dataidx_map = {}

    if partition == "homo":
        total_num = n
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, num_users)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_users)}

    elif partition == "hetero":
        min_size = 0
        N = y.shape[0]
        logging.info("N train = " + str(N))

        idx_batch = []
        while min_size < 10:
            idx_batch = [[] for _ in range(num_users)]
            # for each class in the dataset
            for k in range(num_classes):
                idx_k = np.where(y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y, net_dataidx_map)

    return net_dataidx_map, traindata_cls_counts


def get_partition_indices_test(X_test, y_test, num_classes, num_users, traindata_cls_counts: dict = None):
    net_dataidx_map = defaultdict(list)

    label_indices = get_label_indices(y_test)

    idx = {l: 0 for l in range(num_classes)}  # tracks start of indices for each label used in sampling
    testdata_cls_counts = defaultdict(dict)
    for user in range(num_users):
        user_sampled_labels = range(num_classes) if traindata_cls_counts is None else list(
            traindata_cls_counts[user].keys())
        for label in user_sampled_labels:
            num_samples = int(len(label_indices[label]) / num_users)
            assert num_samples + idx[label] <= len(label_indices[label])
            net_dataidx_map[user].extend(label_indices[label][idx[label]: idx[label] + num_samples])
            testdata_cls_counts[user][label] = num_samples
            idx[label] += num_samples

    return net_dataidx_map, testdata_cls_counts


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


# generate the non-IID distribution for all methods
def read_data_distribution(filename):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def load_partition_data(data_dir, dataset, global_dataloaders, get_dataloader_test, partition_method, partition_alpha,
                        client_number, batch_size,
                        silo_proc_num=0):
    X_train, y_train, X_test, y_test = dataset
    class_num = len(np.unique(y_train))

    train_user_dataidx_map, traindata_cls_counts = get_partition_indices_train(X_train, y_train,
                                                                               partition=partition_method,
                                                                               num_classes=class_num,
                                                                               num_users=client_number,
                                                                               alpha=partition_alpha)

    test_user_dataidx_map, testdata_cls_counts = get_partition_indices_test(X_test, y_test, num_classes=class_num,
                                                                            num_users=client_number,
                                                                            traindata_cls_counts=traindata_cls_counts)

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(train_user_dataidx_map[r]) for r in range(client_number)])
    logging.info("testdata_cls_counts = " + str(testdata_cls_counts))
    test_data_num = sum([len(test_user_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = global_dataloaders
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs_train = train_user_dataidx_map[client_idx]
        dataidxs_test = test_user_dataidx_map[client_idx]
        local_train_number = len(dataidxs_train)
        local_test_number = len(dataidxs_test)
        data_local_num_dict[client_idx] = local_train_number
        logging.info("client_idx = %d, local_train_number = %d, local_test_number = %d" % (
            client_idx, local_train_number, local_test_number))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_test(dataset, data_dir, batch_size, batch_size,
                                                                dataidxs_train, dataidxs_test)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def load_partition_data_distributed(data_dir, dataset, global_dataloaders, get_dataloader_test, process_id,
                                    partition_method, partition_alpha,
                                    client_number, batch_size):
    X_train, y_train, X_test, y_test = dataset
    class_labels = np.unique(y_train)
    class_num = len(class_labels)

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
        train_data_global, test_data_global = global_dataloaders
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
        local_test_num = len(test_dataidxs)

        logging.info("rank = %d, local_train_number = %d, local_test_number = %d" % (
            process_id, local_data_num, local_test_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_test(dataset, data_dir, batch_size, batch_size,
                                                                train_dataidxs, test_dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num
