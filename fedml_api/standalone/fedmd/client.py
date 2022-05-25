from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset

from fedml_api.standalone.utils.BaseClient import BaseClient
import torch
import numpy as np


class Client(BaseClient):
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

    def share_data(self, share_percentage, args):
        local_training_set = self.local_training_data.dataset
        number_to_share = int(share_percentage * len(local_training_set))
        share_idx = torch.from_numpy(np.random.choice(len(local_training_set), size=(number_to_share,), replace=False))
        shared_data = torch.utils.data.Subset(local_training_set, share_idx)
        shared_data = DataLoader(shared_data, batch_size=args.batch_size)
        return shared_data
