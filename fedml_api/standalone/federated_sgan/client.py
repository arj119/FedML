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

    def pre_train(self):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(private_data=self.local_training_data, device=self.device, args=self.args)
