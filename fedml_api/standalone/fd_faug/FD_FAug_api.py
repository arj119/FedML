import copy
import logging
from collections import Counter
from typing import List, Tuple

import torch
import wandb
from torchvision.utils import make_grid
import torchvision.transforms as tfs

from fedml_api.standalone.fd_faug.client import Client
from fedml_api.standalone.fd_faug.model_trainer import FDFAugModelTrainer
from fedml_api.standalone.fd_faug.utils.data_utils import AugmentDataset
from fedml_api.standalone.fedgan.ac_gan_model_trainer import ACGANModelTrainer
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI
from fedml_api.standalone.utils.plot import plot_label_distributions


class FDFAugAPI(HeterogeneousModelBaseTrainerAPI):
    def __init__(self, dataset, device, args, generator, discriminator,
                 client_models: List[Tuple[torch.nn.Module, int]]):
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

        public_datasets = []
        public_dataset_size = 0

        logging.info('############ Creating Augmented dataset (FAug) ############')
        self.mean = torch.Tensor([0.5])
        self.std = torch.Tensor([0.5])

        c: Client
        for c in self.client_list:
            client_shared_data = c.share_data(args.share_percentage, args)
            public_dataset_size += len(client_shared_data.dataset)
            public_datasets.append(client_shared_data)

        self.augmented_data = AugmentDataset(public_datasets)

        self.augment_generator_trainer: ACGANModelTrainer = self._federated_augmentation(self.augmented_data, generator,
                                                                                         discriminator)

        self.generator_model = self.augment_generator_trainer.generator
        # For logging GAN progress
        self.fixed_labels = self.generator_model.generate_balanced_labels(
            self.generator_model.num_classes * 8,
            device='cpu')
        self.fixed_noise = self.generator_model.generate_noise_vector(self.generator_model.num_classes * 8,
                                                                      device='cpu')

        self.log_gan_images('FAug Generator Output', 0)
        wandb.log({'Shared Data Size': public_dataset_size})

        for c in self.client_list:
            client_augmented_data = self.augment_generator_trainer.generate_fake_dataset(10000, 'cpu',
                                                                                         self.args.batch_size)
            c.augment_training_data(augment_data=client_augmented_data)

        logging.info(f'Public dataset size = {public_dataset_size}')
        logging.info('############ Creating public dataset(END) ############')

        self._plot_client_training_data_distribution()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
                       client_models):
        logging.info("############setup_clients (START)#############")

        c_idx = 0
        for local_model, freq in client_models:
            for i in range(freq):
                model_trainer = FDFAugModelTrainer(copy.deepcopy(local_model))
                c = Client(c_idx, train_data_local_dict[c_idx], test_data_local_dict[c_idx],
                           train_data_local_num_dict[c_idx], self.test_global, self.args, self.device,
                           model_trainer)
                self.client_list.append(c)
                c_idx += 1

        logging.info("############setup_clients (END)#############")

    def train(self):
        # FAug
        # TODO: Implement federated augmentation scheme

        for round_idx in range(self.args.comm_round):

            logging.info(f"################Communication round : {round_idx}\n")

            client_subset = self._client_sampling(round_idx)

            logging.info(f"1. Global ensembling phase")
            global_label_logits = dict()
            local_logit_dict = dict()

            client: Client
            for client in client_subset:
                # Communication: Each party computes the class scores on the public dataset, and transmits the result
                # to a central server
                logging.info(f'### Training Client {client.client_idx}')
                client_label_average_logits: dict = client.train()

                local_logit_dict[client.client_idx] = client_label_average_logits

                for label, logit in client_label_average_logits.items():
                    global_label_logits[label] = global_label_logits.get(label, torch.zeros_like(logit)) + logit

            logging.info(f'Updating global average logits and sending to clients')
            M = len(client_subset)

            for client in client_subset:
                # Send ensemble logit vector back to client
                logging.info(f'### Sending to Client {client.client_idx}')
                new_client_label_average_logits = dict()
                for label, sum_logits in global_label_logits.items():
                    new_client_label_average_logits[label] = (sum_logits - local_logit_dict[client.client_idx].get(
                        label, torch.zeros_like(sum_logits))) / (M - 1)

                # Return updated global logits back to client
                logging.info(f'Sent to client: {client.client_idx}')
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

    def _plot_client_training_data_distribution(self):
        columns = ['Client Idx', 'Sample Number', 'Training Dataset Size', 'Test Dataset Size']
        table_data = [[c.client_idx, c.local_sample_number, c.get_dataset_size('train'), c.get_dataset_size('test')] for
                      c in
                      self.client_list]
        wandb.log({'Client Dataset Size Distribution': wandb.Table(columns=columns, data=table_data)})

        # Train
        client_label_counts = [c.get_label_distribution(mode='train') for c in self.client_list]
        client_training_label_count = {client_idx: label_count for client_idx, label_count in client_label_counts}

        # Add public dataset
        train_classes = list(torch.concat([label for _, label in self.augmented_data], dim=0).numpy())
        client_training_label_count[-1] = dict(Counter(train_classes))

        plot_label_distributions(client_training_label_count, self.class_num, alpha=self.args.partition_alpha,
                                 dataset='Train')

        # Test
        client_label_counts = [c.get_label_distribution(mode='test') for c in self.client_list]
        client_test_label_count = {client_idx: label_count for client_idx, label_count in client_label_counts}
        plot_label_distributions(client_test_label_count, self.class_num, alpha=self.args.partition_alpha,
                                 dataset='Test')

    def _federated_augmentation(self, train_data, generator, discriminator):
        logging.info('############ FAug ############')
        trainer = ACGANModelTrainer(generator, discriminator)

        generator, discriminator = generator.to(self.device), discriminator.to(self.device)

        optimiser_G = trainer.get_client_optimiser(generator, self.args.gen_optimizer, self.args.gen_lr)
        optimiser_D = trainer.get_client_optimiser(discriminator, self.args.client_optimizer, self.args.lr)

        trainer._gan_training(generator, discriminator, train_data, self.args.faug_epochs, optimiser_G, optimiser_D,
                              self.device)
        logging.info('############ FAug Complete ############')
        return trainer

    def log_gan_images(self, caption, round_idx):
        generator = self.generator_model.to(self.device)
        images = make_grid(
            self.denorm(generator(self.fixed_noise.to(self.device), self.fixed_labels.to(self.device))),
            nrow=8,
            padding=2,
            normalize=False,
            range=None,
            scale_each=False, pad_value=0)
        images = wandb.Image(images, caption=caption)
        wandb.log({f"Generator Outputs": images, 'Round': round_idx})

    def denorm(self, x, channels=None, w=None, h=None, resize=False, device='cpu'):
        unnormalize = tfs.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist()).to(device)
        x = unnormalize(x)
        if resize:
            if channels is None or w is None or h is None:
                print('Number of channels, width and height must be provided for resize.')
            x = x.view(x.size(0), channels, w, h)
        return x
