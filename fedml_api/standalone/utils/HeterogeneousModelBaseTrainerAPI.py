import copy
import logging
import random
from abc import ABC, abstractmethod
import numpy as np
import torch
import wandb

from fedml_api.standalone.utils.plot import plot_label_distributions


class HeterogeneousModelBaseTrainerAPI(ABC):
    def __init__(self, dataset, device, args):
        """
        Args:
            dataset: Dataset presplit into data loaders
            device: Device to run training on
            args: Additional args
            client_models: List of client models and their frequency participating (assuming a stateful algorithm for simplicity)
        """
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.class_num = class_num
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

    @abstractmethod
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, client_models):
        pass

    @abstractmethod
    def train(self):
        pass

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        global_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client in self.client_list:
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            # train data
            train_local_metrics = client.local_test('train')
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test('test')
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            # global test data
            # global_metrics = client.local_test('val')
            # global_test_metrics['num_samples'].append(copy.deepcopy(global_metrics['test_total']))
            # global_test_metrics['num_correct'].append(copy.deepcopy(global_metrics['test_correct']))
            # global_test_metrics['losses'].append(copy.deepcopy(global_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        # test on global test dataset
        # global_test_acc = sum(global_test_metrics['num_correct']) / sum(global_test_metrics['num_samples'])
        # global_test_loss = sum(global_test_metrics['losses']) / sum(global_test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "Round": round_idx})
        wandb.log({"Train/Loss": train_loss, "Round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "Round": round_idx})
        wandb.log({"Test/Loss": test_loss, "Round": round_idx})
        logging.info(stats)

        # stats = {'global_test_acc': global_test_acc, 'global_test_loss': global_test_loss}
        # wandb.log({"Global Test/Acc": global_test_acc, "Round": round_idx})
        # wandb.log({"Global Test/Loss": global_test_loss, "Round": round_idx})
        # logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, None, self.val_global, None)
        # test data
        test_metrics = client.local_test('val')

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)

    def _plot_client_training_data_distribution(self):
        columns = ['Client Idx', 'Sample Number', 'Training Dataset Size', 'Test Dataset Size']
        table_data = [[c.client_idx, c.local_sample_number, c.get_dataset_size('train'), c.get_dataset_size('test')] for
                      c in
                      self.client_list]
        wandb.log({'Client Dataset Size Distribution': wandb.Table(columns=columns, data=table_data)})

        # Train
        client_label_counts = [c.get_label_distribution(mode='train') for c in self.client_list]
        client_training_label_count = {client_idx: label_count for client_idx, label_count in client_label_counts}
        plot_label_distributions(client_training_label_count, self.class_num, alpha=self.args.partition_alpha,
                                 dataset='Train')

        # Test
        client_label_counts = [c.get_label_distribution(mode='test') for c in self.client_list]
        client_test_label_count = {client_idx: label_count for client_idx, label_count in client_label_counts}
        plot_label_distributions(client_test_label_count, self.class_num, alpha=self.args.partition_alpha,
                                 dataset='Test')
