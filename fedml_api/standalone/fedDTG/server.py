import copy
import logging
import random
from typing import List, Tuple

import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
import torchvision.transforms as tfs

from fedml_api.standalone.fedDTG.ac_gan_model_trainer import ACGANModelTrainer
from fedml_api.standalone.fedDTG.client import FedDTGClient
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI


class FedDTGAPI(HeterogeneousModelBaseTrainerAPI):
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

        self.mean = torch.Tensor([0.5])
        self.std = torch.Tensor([0.5])

        # self.generator = MyModelTrainer(generator)
        # self.discriminator = MyModelTrainer(discriminator)
        self.global_models = ACGANModelTrainer(generator, discriminator, None)
        # For logging GAN progress
        self.fixed_labels = self.global_models.generator.generate_balanced_labels(
            self.global_models.generator.num_classes,
            device='cpu')
        self.fixed_noise = self.global_models.generator.generate_noise_vector(self.global_models.generator.num_classes,
                                                                              device='cpu')

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
                    copy.deepcopy(self.global_models.generator),
                    copy.deepcopy(self.global_models.discriminator),
                    copy.deepcopy(local_model)
                )
                c = FedDTGClient(c_idx, train_data_local_dict[c_idx], test_data_local_dict[c_idx],
                                 train_data_local_num_dict[c_idx], self.test_global, self.args, self.device,
                                 model_trainer)
                c_idx += 1
                self.client_list.append(c)

        logging.info("############setup_clients (END)#############")

    def train(self):
        g_global, d_global = self.global_models.get_model_params()
        class_labels = list(range(self.class_num))
        # kd_batch_size = 128
        noise_size = 10000

        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            # ---------------
            # GAN Training
            # ---------------
            logging.info('########## Gan Training ########')

            g_locals, d_locals = [], []

            client: FedDTGClient
            for idx, client in enumerate(self.client_list):
                # Local round
                w_g, w_d = client.train((copy.deepcopy(g_global), copy.deepcopy(d_global)), round_idx)

                g_locals.append(w_g)
                d_locals.append(w_d)

            # update global weights
            g_global = self._aggregate(g_locals)
            # self.generator.set_model_params(g_global)
            d_global = self._aggregate(d_locals)
            # self.discriminator.set_model_params(d_global)
            self.global_models.set_model_params((g_global, d_global))

            # ---------------
            # Distillation Phase
            # ---------------
            logging.info('########## Distillation ########')

            # noise_vector = self.generator.model.generate_noise_vector(noise_size, device='cpu')
            # labels = self.generator.model.generate_balanced_labels(noise_size, device='cpu')
            # noise_labels = TensorDataset(noise_vector, labels)
            # noise_labels_loader = DataLoader(noise_labels, batch_size=self.args.batch_size)
            noise_vector = self.global_models.generator.generate_noise_vector(noise_size, device=self.device)
            labels = self.global_models.generator.generate_balanced_labels(noise_size, device=self.device)
            noise_labels = TensorDataset(noise_vector, labels)
            noise_labels_loader = DataLoader(noise_labels, batch_size=self.args.batch_size)
            del noise_labels
            synth_data = self.global_models.generate_distillation_dataset(noise_labels_loader, device=self.device)
            distillation_dataset = DataLoader(TensorDataset(synth_data, labels), batch_size=self.args.batch_size)

            local_logits = []
            logging.info("########## Acquiring distillation logits... #########")
            for idx, client in enumerate(self.client_list):
                # logits = client.get_distillation_logits((copy.deepcopy(g_global), copy.deepcopy(d_global)),
                #                                         noise_labels_loader)
                logits = client.get_distillation_logits((copy.deepcopy(g_global), copy.deepcopy(d_global)),
                                                        distillation_dataset)
                local_logits.append(logits)
                logging.info(f"Client {idx} complete")

            # Calculate average soft labels
            # consensus_logits = torch.mean(torch.stack(local_logits), dim=0)
            # consensus_logits_data_loader = DataLoader(TensorDataset(consensus_logits), batch_size=self.args.batch_size)

            logging.info(f"######## Knowledge distillation stage ########")
            for idx, client in enumerate(self.client_list):
                # Calculate teacher logits for client
                logging.info(f"##### Client {idx} #####")
                consensus_logits = torch.mean(torch.stack(local_logits[:idx] + local_logits[idx + 1:]), dim=0)
                consensus_logits_data_loader = DataLoader(TensorDataset(consensus_logits),
                                                          batch_size=self.args.batch_size)
                client.classifier_knowledge_distillation(consensus_logits_data_loader, distillation_dataset)

            if round_idx % 1 == 0:
                logging.info("########## Logging generator images... #########")
                self.log_gan_images(caption=f'Generator Output, communication round: {round_idx}')
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

    def _aggregate(self, w_locals):
        w = 1 / len(w_locals)
        averaged_params = w_locals[0]
        for k in averaged_params.keys():
            for i, local_model_params in enumerate(w_locals):
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def log_gan_images(self, caption):
        images = make_grid(
            self.denorm(
                self.global_models.generator(self.fixed_noise.to(self.device), self.fixed_labels.to(self.device))),
            nrow=8,
            padding=2,
            normalize=False,
            range=None,
            scale_each=False, pad_value=0)
        images = wandb.Image(images, caption=caption)
        wandb.log({f"Generator Outputs": images})

    def denorm(self, x, channels=None, w=None, h=None, resize=False, device='cpu'):
        unnormalize = tfs.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist()).to(device)
        x = unnormalize(x)
        if resize:
            if channels is None or w is None or h is None:
                print('Number of channels, width and height must be provided for resize.')
            x = x.view(x.size(0), channels, w, h)
        return x
