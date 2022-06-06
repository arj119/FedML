import logging
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Subset

from .datasets import EMNIST_truncated

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


def _data_transforms_emnist():
    EMNIST_MEAN = (0.5,)
    EMNIST_STD = (0.5,)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])

    return train_transform, valid_transform


def load_emnist_data(datadir):
    train_transform, test_transform = _data_transforms_emnist()

    mnist_train_ds = EMNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    mnist_test_ds = EMNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    return (X_train, y_train, X_test, y_test)


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_EMNIST(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_EMNIST(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_EMNIST(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = EMNIST_truncated

    transform_train, transform_test = _data_transforms_emnist()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_EMNIST_train_subset(datadir, train_bs, size):
    dl_obj = EMNIST_truncated

    transform_train, transform_test = _data_transforms_emnist()

    train_ds = dl_obj(datadir, train=True, transform=transform_train, download=True)
    train_data_num = len(train_ds)
    sample_indices = random.sample(range(train_data_num), min(size, train_data_num))
    subset = Subset(train_ds, sample_indices)
    return data.DataLoader(subset, batch_size=train_bs, shuffle=True)


def get_dataloader_test_EMNIST(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = EMNIST_truncated

    transform_train, transform_test = _data_transforms_emnist()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def load_partition_data_distributed_emnist(process_id, dataset, data_dir, partition_method, partition_alpha,
                                           client_number, batch_size):
    load_partition_data_distributed(data_dir=data_dir, dataset=load_emnist_data(data_dir),
                                    global_dataloaders=get_dataloader(dataset, data_dir, batch_size, batch_size),
                                    get_dataloader_test=get_dataloader_test,
                                    process_id=process_id,
                                    partition_alpha=partition_alpha,
                                    partition_method=partition_method,
                                    client_number=client_number,
                                    batch_size=batch_size
                                    )


def load_partition_data_emnist(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, r,
                               silo_proc_num=0):
    return load_partition_data(data_dir, dataset=load_emnist_data(data_dir),
                               global_dataloaders=get_dataloader(dataset, data_dir, batch_size, batch_size),
                               get_dataloader_test=get_dataloader_test,
                               partition_method=partition_method,
                               partition_alpha=partition_alpha,
                               client_number=client_number,
                               batch_size=batch_size,
                               r=r,
                               silo_proc_num=silo_proc_num
                               )
