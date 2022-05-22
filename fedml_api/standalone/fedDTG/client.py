import logging

from torch.utils.data import random_split, DataLoader, TensorDataset

from fedml_api.standalone.federated_sgan.model_trainer import FedSSGANModelTrainer
from fedml_api.standalone.utils.BaseClient import BaseClient


class FedDTGClient(BaseClient):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                 device, model_trainer):
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                         device,
                         model_trainer)

        # self.local_training_data, sizes = self.prepare_local_training_data(local_training_data)
        self.local_training_data = local_training_data
        self.distillation_dataset = None

        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        logging.info(f"""
            ########### Client {client_idx}: ###############
            Local sample number: {self.local_sample_number}
            Training dataset size: {self.get_dataset_size('train')}
            Test dataset size: {self.get_dataset_size('test')}
            ################################################
        """)

    def train(self, w_global, communication_round=0):
        logging.info(f'### Training Client {self.client_idx} ###')
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        return self.model_trainer.get_model_params()

    def pre_train(self):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(private_data=self.local_training_data, device=self.device, args=self.args)

    def generate_distillation_set(self, noise_labels_loader):
        distillation_dataset = self.model_trainer.generate_distillation_dataset(noise_labels_loader, self.device)
        self.distillation_dataset = distillation_dataset

    # def get_distillation_logits(self, w_global, noise_labels_loader):
    #     self.model_trainer.set_model_params(w_global)
    #     self.generate_distillation_set(noise_labels_loader)
    #     return self.model_trainer.get_classifier_logits(self.distillation_dataset, self.device)

    def get_distillation_logits(self, w_global, distillation_dataset):
        self.model_trainer.set_model_params(w_global)
        # self.generate_distillation_set(noise_labels_loader)
        return self.model_trainer.get_classifier_logits(distillation_dataset, self.device)

    def classifier_knowledge_distillation(self, consensus_logits, distillation_dataset):
        self.model_trainer.knowledge_distillation(distillation_dataset, consensus_logits, self.device, self.args)
