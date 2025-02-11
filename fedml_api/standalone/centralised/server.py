import copy
import logging
from typing import List, Tuple

import torch

from fedml_api.standalone.centralised.client import CentralisedClient
from fedml_api.standalone.centralised.model_trainer import CentralisedModelTrainer
from fedml_api.standalone.fedmd.utils.data_utils import PublicDataset
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI


class CentralisedAPI(HeterogeneousModelBaseTrainerAPI):
    def __init__(self, dataset, device, args, client_models: List[Tuple[torch.nn.Module, int]]):
        """
        Args:
            dataset: Dataset presplit into data loaders
            device: Device to run training on
            args: Additional args
            client_models: List of client models and their frequency participating (assuming a stateful algorithm for simplicity)
        """
        super().__init__(dataset, device, args)
        c_idx = 0
        public_datasets = []
        for local_model, freq in client_models:
            for i in range(freq):
                public_datasets.append(self.train_data_local_dict[c_idx])
                c_idx += 1

        self.centralised_data = PublicDataset(public_datasets)
        self._setup_clients(self.train_data_local_num_dict, None, self.test_data_local_dict,
                            client_models)

        self._plot_client_training_data_distribution()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
                       client_models):
        logging.info("############setup_clients (START)#############")
        for c_idx, (local_model, _) in enumerate(client_models):
            model_trainer = CentralisedModelTrainer(
                local_model,
                self.args)

            c = CentralisedClient(c_idx, self.centralised_data, test_data_local_dict[c_idx],
                                  len(self.centralised_data),
                                  self.test_global, self.args, self.device,
                                  model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        logging.info('\n###############Pre-Training clients#############\n')
        for i, c in enumerate(self.client_list):
            logging.info(f'Pre=training client: {i}')
            c.pre_train()
        logging.info('###############Pre-Training clients (END)###########\n')
        self._local_test_on_all_clients(-1)

        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))

            for idx, client in enumerate(self.client_list):
                # Local round
                client.train()

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)
