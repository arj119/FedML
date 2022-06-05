from torch.utils.data import DataLoader
import torch
import numpy as np

from fedml_api.standalone.fd_faug.model_trainer import FDFAugModelTrainer
from fedml_api.standalone.fd_faug.utils.data_utils import AugmentDataset
from fedml_api.standalone.utils.BaseClient import BaseClient


class Client(BaseClient):
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                 device,
                 model_trainer: FDFAugModelTrainer):
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                         device,
                         model_trainer)

        self.global_label_logits = None

    def train(self, w_global=None):
        return self.model_trainer.train(self.local_training_data, self.global_label_logits, self.device, self.args)

    def get_logits(self, public_data):
        return self.model_trainer.get_logits(public_data, self.device)

    def update_global_label_logits(self, logits):
        self.global_label_logits = logits

    def augment_training_data(self, augment_data):
        self.local_training_data = AugmentDataset([self.local_training_data, augment_data])

    def local_test(self, data='test'):
        if data == 'test':
            test_data = self.local_test_data
        elif data == 'val':
            test_data = self.global_val_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def share_data(self, share_percentage, args):
        local_training_set = self.local_training_data.dataset
        number_to_share = int(share_percentage * len(local_training_set))
        share_idx = torch.from_numpy(np.random.choice(len(local_training_set), size=(number_to_share,), replace=False))
        shared_data = torch.utils.data.Subset(local_training_set, share_idx)
        shared_data = DataLoader(shared_data, batch_size=args.batch_size)
        return shared_data
