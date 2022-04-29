import logging
from torch.utils.data import DataLoader
from collections import Counter
import torch

class BaseClient:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data: DataLoader = local_training_data
        self.local_test_data: DataLoader = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self._get_training_data_from_tuple()

        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def get_training_label_distribution(self):
        local_training_data = self._get_training_data_from_tuple()
        train_classes = list(torch.concat([label for _, label in local_training_data], dim=0).numpy())
        return self.client_idx, dict(Counter(train_classes))

    def _get_training_data_from_tuple(self):
        return self.local_training_data[0] if isinstance(self.local_training_data,
                                                         tuple) else self.local_training_data
