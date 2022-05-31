import logging

from fedml_api.standalone.fd_faug.model_trainer import FDFAugModelTrainer
from fedml_api.standalone.utils.BaseClient import BaseClient


class Client(BaseClient):
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args, device,
                 model_trainer: FDFAugModelTrainer):
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args, device,
                         model_trainer)

        self.global_label_logits = None

    def train(self, w_global=None):
        return self.model_trainer.train(self.local_training_data, self.global_label_logits, self.device, self.args)

    def get_logits(self, public_data):
        return self.model_trainer.get_logits(public_data, self.device)

    def update_global_label_logits(self, logits):
        self.global_label_logits = logits

    def local_test(self, data='test'):
        if data == 'test':
            test_data = self.local_test_data
        elif data == 'val':
            test_data = self.global_val_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
