import logging

from torch.utils.data import random_split, DataLoader, TensorDataset
import torch
import numpy as np

from fedml_api.standalone.utils.BaseClient import BaseClient


class FedUAGANClient(BaseClient):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                 device, model_trainer):
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                         device,
                         model_trainer)

        # self.local_training_data, sizes = self.prepare_local_training_data(local_training_data)
        self.local_training_data = local_training_data
        self.local_synthetic_data = None

        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        logging.info(f"""
            ########### Client {client_idx}: ###############
            Local sample number: {self.local_sample_number}
            Training dataset size: {self.get_dataset_size('train')}
            Test dataset size: {self.get_dataset_size('test')}
            ################################################
        """)

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

    def train(self, m, D_syn, fake_labels, communication_round=0):
        D_real = self._uniform_sample_dataset(m)
        return self.model_trainer.train(D_real, D_syn, fake_labels, self.device, self.args)

    def pre_train(self):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(private_data=self.local_training_data, device=self.device, args=self.args)

    def update_synthetic_dataset(self):
        target_size = self.get_dataset_size('train') * 5
        num_batches = len(self.local_training_data)
        synthetic_dataset, size, batch_size = self.model_trainer.generate_synthetic_dataset(target_size,
                                                                                            num_batches,
                                                                                            device=self.device)

        logging.info(f'Client: {self.client_idx} created synthetic dataset of size: {size}, batch size {batch_size}')
        self.local_synthetic_data = synthetic_dataset

    def set_synthetic_dataset(self, unlabelled_dataset):
        if unlabelled_dataset is None:
            self.local_synthetic_data = None
            return

        size = len(unlabelled_dataset)
        num_batches = len(self.local_training_data)
        batch_size = size // num_batches if size > num_batches else size
        self.local_synthetic_data = DataLoader(unlabelled_dataset, batch_size=batch_size)

    def generate_synthetic_dataset(self):
        target_size = self.get_dataset_size('train') * 5
        synthetic_dataset, size = self.model_trainer.generate_synthetic_dataset(target_size,
                                                                                device=self.device)
        logging.info(f'Client: {self.client_idx} created synthetic dataset of size: {size}')
        return synthetic_dataset

    def get_discriminator_outputs(self, D_syn):
        return self.model_trainer.get_discriminator_output(D_syn, self.device)


    def _uniform_sample_dataset(self, b_size):
        dataset = self.local_training_data.dataset
        indices = torch.from_numpy(np.random.choice(len(dataset), size=(b_size,), replace=False))
        subset = torch.utils.data.Subset(dataset, indices)
        return DataLoader(subset, batch_size=b_size)
