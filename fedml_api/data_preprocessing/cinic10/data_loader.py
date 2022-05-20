import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import ImageFolderTruncated

# logging.basicConfig()
from ..utils.partition import load_partition_data, load_partition_data_distributed

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cinic10():
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(
                                              lambda x: F.pad(x.unsqueeze(0),
                                                              (4, 4, 4, 4),
                                                              mode='reflect').data.squeeze()),
                                          transforms.ToPILImage(),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=cinic_mean,
                                                               std=cinic_std),
                                          ])

    # Transformer for test set
    valid_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(
                                              lambda x: F.pad(x.unsqueeze(0),
                                                              (4, 4, 4, 4),
                                                              mode='reflect').data.squeeze()),
                                          transforms.ToPILImage(),
                                          transforms.RandomCrop(32),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=cinic_mean,
                                                               std=cinic_std),
                                          ])
    return train_transform, valid_transform


def load_cinic10_data(datadir):
    _train_dir = datadir + str('/train')
    logging.info("_train_dir = " + str(_train_dir))
    _test_dir = datadir + str('/test')
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Lambda(
                                                                                  lambda x: F.pad(x.unsqueeze(0),
                                                                                                  (4, 4, 4, 4),
                                                                                                  mode='reflect').data.squeeze()),
                                                                              transforms.ToPILImage(),
                                                                              transforms.RandomCrop(32),
                                                                              transforms.RandomHorizontalFlip(),
                                                                              transforms.ToTensor(),
                                                                              transforms.Normalize(mean=cinic_mean,
                                                                                                   std=cinic_std),
                                                                              ]))

    testset = ImageFolderTruncated(_test_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Lambda(
                                                                                lambda x: F.pad(x.unsqueeze(0),
                                                                                                (4, 4, 4, 4),
                                                                                                mode='reflect').data.squeeze()),
                                                                            transforms.ToPILImage(),
                                                                            transforms.RandomCrop(32),
                                                                            transforms.RandomHorizontalFlip(),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(mean=cinic_mean,
                                                                                                 std=cinic_std),
                                                                            ]))
    X_train, y_train = trainset.imgs, trainset.targets
    X_test, y_test = testset.imgs, testset.targets
    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    X_train, y_train, X_test, y_test = load_cinic10_data(datadir)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    n_train = len(X_train)
    # n_test = len(X_test)

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CINIC10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CINIC10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_cinic10(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs, transform=transform_train)
    test_ds = dl_obj(valdir, transform=transform_train)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def get_dataloader_test_cinic10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = ImageFolderTruncated

    transform_train, transform_test = _data_transforms_cinic10()

    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'test')

    train_ds = dl_obj(traindir, dataidxs=dataidxs_train, transform=transform_train)
    test_ds = dl_obj(valdir, dataidxs=dataidxs_test, transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def load_partition_data_distributed_cinic10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    load_partition_data_distributed(data_dir=data_dir, dataset=load_cinic10_data(data_dir),
                                    global_dataloaders=get_dataloader(dataset, data_dir, batch_size, batch_size),
                                    get_dataloader_test=get_dataloader_test,
                                    process_id=process_id,
                                    partition_alpha=partition_alpha,
                                    partition_method=partition_method,
                                    client_number=client_number,
                                    batch_size=batch_size
                                    )


def load_partition_data_cinic10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    return load_partition_data(data_dir, dataset=load_cinic10_data(data_dir),
                               global_dataloaders=get_dataloader(dataset, data_dir, batch_size, batch_size),
                               get_dataloader_test=get_dataloader_test,
                               partition_method=partition_method,
                               partition_alpha=partition_alpha,
                               client_number=client_number,
                               batch_size=batch_size,
                               )
