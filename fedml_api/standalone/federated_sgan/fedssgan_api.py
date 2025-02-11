import copy
import logging
import random
from typing import List, Tuple

import numpy as np
import torch
import wandb
from torch.utils.data import ConcatDataset

from fedml_api.standalone.fedavg.my_model_trainer import MyModelTrainer
from fedml_api.standalone.federated_sgan.ac_gan_model_trainer import ACGANModelTrainer
from fedml_api.standalone.federated_sgan.client import FedSSGANClient
from fedml_api.standalone.federated_sgan.model_trainer import FedSSGANModelTrainer
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI


class FedSSGANAPI(HeterogeneousModelBaseTrainerAPI):
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
                model_trainer = ACGANModelTrainer(
                    copy.deepcopy(self.global_model.model),
                    copy.deepcopy(local_model)
                )
                c = FedSSGANClient(c_idx, train_data_local_dict[c_idx], test_data_local_dict[c_idx],
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

        unlabelled_synthesised_data = None
        w_global = self.global_model.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            synthesised_data_locals = []
            client_synthesised_data_lens = {'round': round_idx}

            client: FedSSGANClient
            for idx, client in enumerate(self.client_list):
                # Update client synthetic datasets
                # client.set_synthetic_dataset(unlabelled_synthesised_data)

                # Local round
                w = client.train(copy.deepcopy(w_global), round_idx)
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            #     synthetic_data = client.generate_synthetic_dataset()
            #     if synthetic_data is not None:
            #         synthesised_data_locals.append(synthetic_data)
            #         client_synthesised_data_lens[f'Client_{idx}: Synthetic Dataset Size'] = len(synthetic_data)
            #     else:
            #         client_synthesised_data_lens[f'Client_{idx}: Synthetic Dataset Size'] = 0
            #
            # if len(synthesised_data_locals) > 0:
            #     unlabelled_synthesised_data = ConcatDataset(synthesised_data_locals)
            #     logging.info(f'\n Synthetic Unlabelled Dataset Size: {len(unlabelled_synthesised_data)}\n')
            #     client_synthesised_data_lens['Total Synthetic Dataset Size'] = len(unlabelled_synthesised_data)
            # else:
            #     unlabelled_synthesised_data = None
            #     client_synthesised_data_lens['Total Synthetic Dataset Size'] = 0

            # wandb.log(client_synthesised_data_lens)

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
