import logging

from fedml_api.standalone.fd_faug.model_trainer import FDModelTrainer


class Client:
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer: FDModelTrainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.global_label_logits = None


    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self):
        return self.model_trainer.train(self.local_training_data, self.global_label_logits, self.device, self.args)

    def get_logits(self, public_data):
        return self.model_trainer.get_logits(public_data, self.device)

    def update_global_label_logits(self, logits):
        self.global_label_logits = logits

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
