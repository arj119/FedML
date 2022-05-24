import copy
import logging
from typing import List, Tuple

import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
import torch.nn as nn
import torchvision.transforms as tfs

from fedml_api.model.cv.generator import Generator
from fedml_api.standalone.federated_uagan.ac_gan_model_trainer import ACGANModelTrainer
from fedml_api.standalone.federated_uagan.client import FedUAGANClient
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI

"""
Implementing UA-GAN idea https://arxiv.org/pdf/2102.04655.pdf
"""


class FedUAGANAPI(HeterogeneousModelBaseTrainerAPI):
    def __init__(self, dataset, device, args, generator, client_models: List[Tuple[torch.nn.Module, int]]):
        """
        Args:
            dataset: Dataset presplit into data loaders
            device: Device to run training on
            args: Additional args
            client_models: List of client models and their frequency participating (assuming a stateful algorithm for simplicity)
        """
        super().__init__(dataset, device, args)

        self.generator: Generator = generator

        # For logging GAN progress
        self.fixed_labels = self.generator.generate_balanced_labels(self.generator.num_classes, device='cpu')
        self.fixed_noise = self.generator.generate_noise_vector(self.generator.num_classes, device='cpu')

        self.total_num_samples = self._setup_clients(self.train_data_local_num_dict, self.train_data_local_dict,
                                                     self.test_data_local_dict,
                                                     client_models)

        self.weights = self._compute_weights()
        self.real_label, self.fake_label = 1, 0

        self.mean = torch.Tensor([0.5])
        self.std = torch.Tensor([0.5])

        self._plot_client_training_data_distribution()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
                       client_models):
        logging.info("############setup_clients (START)#############")

        total_num_samples = 0
        c_idx = 0
        for local_model, freq in client_models:
            for i in range(freq):
                model_trainer = ACGANModelTrainer(
                    copy.deepcopy(self.generator),
                    copy.deepcopy(local_model),
                    self.args
                )
                c = FedUAGANClient(c_idx, train_data_local_dict[c_idx], test_data_local_dict[c_idx],
                                   train_data_local_num_dict[c_idx], self.test_global, self.args, self.device,
                                   model_trainer)
                c_idx += 1
                total_num_samples += c.get_sample_number()
                self.client_list.append(c)

        logging.info("############setup_clients (END)#############")
        return total_num_samples

    def train(self):
        m = self.args.batch_size

        optimiser_G = self._get_optimiser()

        label_real_adv = torch.full((m, 1), self.real_label, device=self.device,
                                    requires_grad=False, dtype=torch.float)

        adversarial_loss = nn.BCELoss().to(self.device)
        auxiliary_loss = nn.CrossEntropyLoss().to(self.device)

        for round_idx in range(self.args.comm_round):
            logging.info("\n################Communication round : {}".format(round_idx))

            """
                Train Client Discriminators
            """
            logging.info("########## Training Discriminators #########")
            for e in range(self.args.discriminator_epochs):
                # G generates synthetic data D_{syn}
                D_syn, fake_labels = self.generate_synthetic_dataset(m, self.device)

                # Update client discriminators
                client: FedUAGANClient
                for idx, client in enumerate(self.client_list):
                    # Send D_syn to client sites and update discriminator parameters
                    # Local round
                    errD = client.train(m, D_syn, fake_labels, round_idx)
                    logging.info(f"Client {idx}: Disc Loss: {errD}")
                    wandb.log({f'Client {idx} Disc/Loss': errD, 'round': round_idx})

            logging.info("########## Training Generator #########")

            # Update generator
            optimiser_G.zero_grad()

            del D_syn
            D_syn, fake_labels = self.generate_synthetic_dataset(m, self.device)

            D_local_outs = []  # Stores tuple of (client_sample_number, discriminator_gan_output, discriminator_class_output)

            client: FedUAGANClient
            for client in self.client_list:
                D_gan_output, D_cls_logits = client.get_discriminator_outputs(D_syn)
                D_local_outs.append((client.client_idx, D_gan_output, D_cls_logits))

            D_ua_gan_output, D_ua_cls_logits = self._calculate_D_ua(D_local_outs, fake_labels)

            logging.info(f'D_ua_gan_output {D_ua_gan_output.shape}, \t D_ua_cls_logits {D_ua_cls_logits.shape}')

            errG = (adversarial_loss(D_ua_gan_output, label_real_adv) + auxiliary_loss(D_ua_cls_logits,
                                                                                       fake_labels)) / 2

            errG.backward()
            optimiser_G.step()

            wandb.log({'Gen/Loss': errG.item(), 'round': round_idx})

            del D_syn
            logging.info(f'Generator loss: {errG.item()}')

            if round_idx % 20 == 0:
                logging.info("########## Logging generator images... #########")
                self.log_gan_images(caption=f'Generator Output, communication round: {round_idx}', round_idx=round_idx)
                logging.info("########## Logging generator images... Complete #########")

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

    def log_gan_images(self, caption, round_idx):
        images = make_grid(
            self.denorm(self.generator(self.fixed_noise.to(self.device), self.fixed_labels.to(self.device))), nrow=8,
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

    def generate_synthetic_dataset(self, batch_size, device):
        generator = self.generator.to(device)
        generator.eval()

        labels = generator.generate_balanced_labels(batch_size, device=device)
        noise = generator.generate_noise_vector(batch_size, device=device)
        generated_images = generator(noise, labels)

        dataset = TensorDataset(generated_images, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader, labels

    """
    Inspired from https://github.com/yz422/UAGAN/blob/d3248f320f76a1dd27b89c4c193d11f411d54e9e/models/uagan_model.py#L126
    """

    def _get_optimiser(self):
        if self.args.client_optimizer == "sgd":
            optimiser_G = torch.optim.SGD(self.generator.parameters(), lr=self.args.lr)

        else:
            beta1, beta2 = 0.5, 0.999
            optimiser_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()),
                                           lr=self.args.lr,
                                           weight_decay=self.args.wd,
                                           amsgrad=True,
                                           betas=(beta1, beta2)
                                           )

        return optimiser_G

    def _calculate_D_ua(self, D_local_outs, labels):
        """

        Args:
            D_local_outs: List of tuples of client discriminator outputs i.e. (client_idx, d_i_gan_out, d_i_cls_logits)
            labels: Fake labels to match to

        Returns:

        """
        D_ua_gan_out = None
        D_ua_cls_logits = None
        weights = self.weights.to(self.device)
        for (client_idx, d_i_gan_out, d_i_cls_logits) in D_local_outs:
            weight = weights[client_idx][labels]
            # Aggregate odds of individual discriminator outputs
            if D_ua_gan_out is None:
                # Implicitly means D_ua_cls_logits is None too
                D_ua_gan_out = self._calculate_odds(d_i_gan_out) * weight.unsqueeze(1)
                D_ua_cls_logits = d_i_cls_logits * weight.unsqueeze(1)
            else:
                D_ua_gan_out += self._calculate_odds(d_i_gan_out) * weight.unsqueeze(1)
                D_ua_cls_logits = d_i_cls_logits * weight.unsqueeze(1)

        # our aggregation: Convert back into output
        D_ua_gan_out = D_ua_gan_out / (1 + D_ua_gan_out)
        return D_ua_gan_out, D_ua_cls_logits

    def _calculate_odds(self, d_out):
        return d_out / (1 - d_out + 1e-8)

    def _compute_weights(self):
        num_of_labels = torch.zeros((len(self.client_list), self.class_num))
        c: FedUAGANClient
        for c in self.client_list:
            client_idx, label_counts = c.get_label_distribution(mode='train')
            for k in range(self.class_num):
                num_of_labels[client_idx][k] = label_counts.get(k, 0)

        return num_of_labels / num_of_labels.sum(dim=0)
