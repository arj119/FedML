import copy
import logging
import random
from typing import List, Tuple

import numpy as np
import torch
import wandb

from fedml_api.standalone.fedavg.my_model_trainer import MyModelTrainer
from fedml_api.standalone.federated_arjun.client import FedArjunClient
from fedml_api.standalone.federated_arjun.model_trainer import FedArjunModelTrainer
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI


class FedArjunAPI(HeterogeneousModelBaseTrainerAPI):
    def __init__(self, dataset, device, args, adapter_model, client_models: List[Tuple[torch.nn.Module, int]]):
        """
        Args:
            dataset: Dataset presplit into data loaders
            device: Device to run training on
            args: Additional args
            client_models: List of client models and their frequency participating (assuming a stateful algorithm for simplicity)
        """
        super().__init__(dataset, device, args)

        self.global_model = MyModelTrainer(adapter_model)

        self._setup_clients(self.train_data_local_num_dict, self.train_data_local_dict, self.test_data_local_dict,
                            client_models)

        self._plot_client_training_data_distribution()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
                       client_models):
        logging.info("############setup_clients (START)#############")

        c_idx = 0
        for local_model, freq in client_models:
            for i in range(freq):
                model_trainer = FedArjunModelTrainer(
                    copy.deepcopy(self.global_model.model),
                    copy.deepcopy(local_model),
                    self.args)
                c = FedArjunClient(c_idx, train_data_local_dict[c_idx], test_data_local_dict[c_idx],
                                   train_data_local_num_dict[c_idx], self.test_global, self.args, self.device,
                                   model_trainer)
                c_idx += 1
                self.client_list.append(c)

        logging.info("############setup_clients (END)#############")

    def train(self):
        logging.info('\n###############Pre-Training clients#############\n')
        for i, c in enumerate(self.client_list):
            logging.info(f'Pre=training client: {i}')
            c.pre_train()
        logging.info('###############Pre-Training clients (END)###########\n')

        w_global = self.global_model.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            for client in self._client_sampling(round_idx):
                # Local round
                w = client.train(copy.deepcopy(w_global))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(w_locals)
            self.global_model.set_model_params(w_global)

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
