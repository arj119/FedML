import logging

from torch.utils.data import random_split, DataLoader
import torch
import numpy as np

from fedml_api.standalone.fd_faug.utils.data_utils import AugmentDataset
from fedml_api.standalone.utils.BaseClient import BaseClient


class BaselineClient(BaseClient):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                 device, model_trainer):
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                         device,
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

    def train(self, w_global=None):
        logging.info(f'### Training Client {self.client_idx} ###')
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        logging.info(f'### Training Client {self.client_idx} (complete) ###')

    def pre_train(self):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(private_data=self.local_training_data, device=self.device, args=self.args)

    def augment_training_data(self, augment_data):
        self.local_training_data = AugmentDataset([self.local_training_data, augment_data])

    def share_data(self, share_percentage, args):
        local_training_set = self.local_training_data.dataset
        number_to_share = int(share_percentage * len(local_training_set))
        share_idx = torch.from_numpy(np.random.choice(len(local_training_set), size=(number_to_share,), replace=False))
        shared_data = torch.utils.data.Subset(local_training_set, share_idx)
        shared_data = DataLoader(shared_data, batch_size=args.batch_size)
        return shared_data
