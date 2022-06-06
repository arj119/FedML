from torch.utils.data import DataLoader


def itr_merge(itrs):
    for itr in itrs:
        for v in itr:
            yield v


class PublicDataset:
    def __init__(self, data_loaders):
        self.data_loaders = data_loaders

    def __iter__(self):
        return itr_merge(self.data_loaders)

    def __len__(self):
        size = 0
        for dl in self.data_loaders:
            if isinstance(dl, DataLoader):
                dl_size = len(dl.dataset)
            else:
                dl_size = len(dl)

            size += dl_size
        return size
