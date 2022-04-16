import logging

from fedml_api.standalone.fedmd.model_trainer import FedMLModelTrainer


class Client:
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer: FedMLModelTrainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
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

    def train(self, public_data, consensus_logits):
        self.model_trainer.train(self.local_training_data, self.device, self.args, public_data, consensus_logits)

    def pre_train(self, public_data):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(public_data=public_data, private_data=self.local_training_data, device=self.device,
                                     args=self.args)

    def get_logits(self, public_data):
        return self.model_trainer.get_logits(public_data, self.device)

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
