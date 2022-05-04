import copy
import logging
import random
from typing import List, Tuple

import numpy as np
import torch
import wandb

from fedml_api.standalone.fedmd.client import Client
from fedml_api.standalone.fedmd.model_trainer import FedMLModelTrainer
from fedml_api.standalone.fedmd.utils.data_utils import PublicDataset
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI


class FedMDAPI(HeterogeneousModelBaseTrainerAPI):
    def __init__(self, dataset, device, args, client_models: List[Tuple[torch.nn.Module, int]]):
        """
        Args:
            dataset: Dataset presplit into data loaders
            device: Device to run training on
            args: Additional args
            client_models: List of client models and their frequency participating (assuming a stateful algorithm for simplicity)
        """
        super().__init__(dataset, device, args)

        # Select a sufficient number of data samples to be in public dataset
        if not args.public_dataset_size:
            args.public_dataset_size = 5000

        public_datasets = []
        public_dataset_size = 0
        self.start_local_client_idx = 0

        logging.info('############ Creating public dataset ############')
        for data_loader in self.train_data_local_dict.values():
            public_dataset_size += len(data_loader.dataset)
            public_datasets.append(data_loader)
            if public_dataset_size >= args.public_dataset_size:
                break

        self.public_data = PublicDataset(public_datasets)
        logging.info(f'Public dataset size = {public_dataset_size}')
        logging.info('############ Creating public dataset(END) ############')

        self._setup_clients(self.train_data_local_num_dict, self.train_data_local_dict, self.test_data_local_dict,
                            client_models)

        self._plot_client_training_data_distribution()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
                       client_models):
        logging.info("############setup_clients (START)#############")

        for client_idx, (model, freq) in enumerate(client_models):
            for i in range(freq):
                model_trainer = FedMLModelTrainer(copy.deepcopy(model))
                c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                           train_data_local_num_dict[client_idx], self.test_global, self.args, self.device, model_trainer)
                self.client_list.append(c)

        logging.info("############setup_clients (END)#############")

    def train(self):
        # Transfer learning
        logging.info('\n###############Pre-Training clients#############\n')
        for i, c in enumerate(self.client_list):
            logging.info(f'Pre=training client: {i}')
            c.pre_train(self.public_data)
        logging.info('###############Pre-Training clients (END)###########\n')


        for round_idx in range(self.args.comm_round):

            logging.info(f"################Communication round : {round_idx}\n")

            logging.info(f"1. Communication Start")
            local_logits = []
            for idx, client in enumerate(self.client_list):
                # Communication: Each party computes the class scores on the public dataset, and transmits the result
                # to a central server
                logits = client.get_logits(self.public_data)
                logging.info(f'Retrieving client {idx} logits: {logits.shape}')
                local_logits.append(logits)
            logging.info(f"1. Communication End")

            logging.info(f"2. Aggregate Start")
            # Aggregate: The server computes an updated consensus, which is an average
            consensus_logits = torch.mean(torch.stack(local_logits), dim=0)
            logging.info(f"2. Aggregate End: Consensus logits dims{consensus_logits.shape}")

            logging.info(f"3. Digest + Revisit Start")
            # Distribute, Digest and Revisit
            for idx, client in enumerate(self.client_list):
                logging.info(f'\nTraining client: {idx}\n')
                client.train(self.public_data, consensus_logits)
            logging.info(f"3. Digest + Revisit End")

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

