import copy
import logging
import random
from typing import List, Tuple

import numpy as np
import torch
import wandb

from fedml_api.standalone.fd_faug.client import Client
from fedml_api.standalone.fd_faug.model_trainer import FDFAugModelTrainer
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI


class FDFAugAPI(HeterogeneousModelBaseTrainerAPI):
    def __init__(self, dataset, device, args, client_models: List[Tuple[torch.nn.Module, int]]):
        """
        Args:
            dataset: Dataset presplit into data loaders
            device: Device to run training on
            args: Additional args
            client_models: List of client models and their frequency participating (assuming a stateful algorithm for simplicity)
        """
        super().__init__(dataset, device, args)

        self._setup_clients(self.train_data_local_num_dict, self.train_data_local_dict, self.test_data_local_dict,
                            client_models)

        self._plot_client_training_data_distribution()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
                       client_models):
        logging.info("############setup_clients (START)#############")

        for client_idx, (model, freq) in enumerate(client_models):
            for i in range(freq):
                model_trainer = FDFAugModelTrainer(model)
                c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                           train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
                self.client_list.append(c)

        logging.info("############setup_clients (END)#############")

    def train(self):
        # FAug
        # TODO: Implement federated augmentation scheme

        for round_idx in range(self.args.comm_round):

            logging.info(f"################Communication round : {round_idx}\n")

            logging.info(f"1. Global ensembling phase")
            global_label_logits = dict()
            local_logit_dict = dict()
            for idx, client in enumerate(self.client_list):
                # Communication: Each party computes the class scores on the public dataset, and transmits the result
                # to a central server
                client_label_average_logits: dict = client.train()
                # logging.info(f'Retrieving client {idx} logits: {client_label_average_logits}')

                local_logit_dict[idx] = client_label_average_logits

                for label, logit in client_label_average_logits.items():
                    global_label_logits[label] = global_label_logits.get(label, torch.zeros_like(logit)) + logit

            logging.info(f'Updating global average logits and sending to clients')
            M = len(self.client_list)

            for idx, client in enumerate(self.client_list):
                # Send ensemble logit vector back to client
                new_client_label_average_logits = dict()
                for label, sum_logits in global_label_logits.items():
                    new_client_label_average_logits[label] = (sum_logits - local_logit_dict[idx][label]) / (M - 1)

                # Return updated global logits back to client
                logging.info(f'Sent to client: {idx}')
                client.update_global_label_logits(new_client_label_average_logits)

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

