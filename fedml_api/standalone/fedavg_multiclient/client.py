import logging

from torch.utils.data import random_split, DataLoader, TensorDataset

from fedml_api.standalone.fedavg_multiclient.model_trainer import FedAvgMultiClientModelTrainer
from fedml_api.standalone.federated_arjun.model_trainer import FedArjunModelTrainer
from fedml_api.standalone.utils.BaseClient import BaseClient


class FedAvgMultiClient(BaseClient):

    def pre_train(self):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(private_data=self.local_training_data, device=self.device, args=self.args)
