import logging

from torch.utils.data import random_split, DataLoader, TensorDataset

from fedml_api.standalone.federated_arjun.model_trainer import FedArjunModelTrainer
from fedml_api.standalone.utils.BaseClient import BaseClient


class FedArjunClient(BaseClient):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer: FedArjunModelTrainer):
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                         model_trainer)

        # self.local_training_data, sizes = self.prepare_local_training_data(local_training_data)
        self.local_training_data = local_training_data

        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        logging.info(f"""
            ########### Client {client_idx}: ###############
            Local sample number: {self.local_sample_number}
            Training dataset size: {len(local_training_data)}
            Test dataset size: {len(local_test_data)}
            ################################################
        """)

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data, _ = self.prepare_local_training_data(local_training_data)
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def prepare_local_training_data(self, local_training_data):
        if isinstance(local_training_data, list):
            dataset = TensorDataset(local_training_data[0][0], local_training_data[0][1])
            local_training_data = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=2)

        len_local_dataset = len(local_training_data.dataset)
        transfer_set_size = int(self.args.transfer_set_percentage * len_local_dataset)
        transfer, train = random_split(local_training_data.dataset,
                                       [transfer_set_size, len_local_dataset - transfer_set_size])
        return (DataLoader(train, batch_size=self.args.batch_size),
                DataLoader(transfer, batch_size=self.args.batch_size)), \
               (len_local_dataset - transfer_set_size, transfer_set_size)

    def pre_train(self):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(private_data=self.local_training_data, device=self.device, args=self.args)

