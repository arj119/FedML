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
        length = 0
        for dl in self.data_loaders:
            length += len(dl.dataset)
        return length
