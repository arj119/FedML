import logging

from torch.utils.data import random_split, DataLoader, TensorDataset

from fedml_api.standalone.federated_sgan.model_trainer import FedSSGANModelTrainer
from fedml_api.standalone.utils.BaseClient import BaseClient


class FedSSGANClient(BaseClient):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                 device,
                 model_trainer: FedSSGANModelTrainer):
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

    def train(self, w_global, communication_round=0):
        logging.info(f'### Training Client {self.client_idx} ###')
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train((self.local_training_data, self.local_synthetic_data), self.device, self.args)
        weights = self.model_trainer.get_model_params()
        self.model_trainer.log_gan_images(
            caption=f'Client {self.client_idx}, communication round: {communication_round}',
            client_id=self.client_idx)
        logging.info(f'### Training Client {self.client_idx} (complete) ###')
        return weights

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
