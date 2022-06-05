from fedml_api.standalone.fedmd.utils.data_utils import PublicDataset


class AugmentDataset(PublicDataset):
    def __init__(self, data_loaders):
        super().__init__(data_loaders)
