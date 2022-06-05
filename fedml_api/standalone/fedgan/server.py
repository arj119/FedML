import copy
import logging

import torch
import wandb
from torchvision.utils import make_grid
import torchvision.transforms as tfs

from FID.FIDScorer import FIDScorer
from fedml_api.standalone.fedgan.ac_gan_model_trainer import ACGANModelTrainer
from fedml_api.standalone.utils.BaseClient import BaseClient
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI


class FedGANAPI(HeterogeneousModelBaseTrainerAPI):
    def __init__(self, dataset, device, args, generator, discriminator, num_clients):
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

        self.global_models = ACGANModelTrainer(generator, discriminator)
        # For logging GAN progress
        self.fixed_labels = self.global_models.generator.generate_balanced_labels(
            self.global_models.generator.num_classes,
            device='cpu')
        self.fixed_noise = self.global_models.generator.generate_noise_vector(self.global_models.generator.num_classes,
                                                                              device='cpu')

        self._setup_clients(self.train_data_local_num_dict, self.train_data_local_dict, self.test_data_local_dict,
                            num_clients)

        self._plot_client_training_data_distribution()

        # Generate dataset that can be used to calculate FID score
        self.FID_source_set = self._generate_train_subset(num_samples=10000)
        self.FIDScorer = FIDScorer()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
                       num_clients):
        logging.info("############setup_clients (START)#############")

        for c_idx in range(num_clients):
            model_trainer = ACGANModelTrainer(
                copy.deepcopy(self.global_models.generator),
                copy.deepcopy(self.global_models.disc_classifier)
            )
            c = BaseClient(c_idx, train_data_local_dict[c_idx], test_data_local_dict[c_idx],
                           train_data_local_num_dict[c_idx], self.test_global, self.args, self.device,
                           model_trainer)
            self.client_list.append(c)

        logging.info("############setup_clients (END)#############")

    def train(self):
        g_global, d_global = self.global_models.get_model_params()

        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            # ---------------
            # GAN Training
            # ---------------
            logging.info('########## Gan Training ########')

            client_subset = self._client_sampling(round_idx)

            g_locals, d_locals = [], []

            for client in client_subset:
                # Local round
                w_g, w_d = client.train((copy.deepcopy(g_global), copy.deepcopy(d_global)))
                g_locals.append((client.get_sample_number(), copy.deepcopy(w_g)))
                d_locals.append((client.get_sample_number(), copy.deepcopy(w_d)))

            # update global weights
            g_global = self._aggregate(g_locals)
            d_global = self._aggregate(d_locals)
            self.global_models.set_model_params((g_global, d_global))

            if round_idx % 1 == 0:
                logging.info("########## Logging generator images... #########")
                self.log_gan_images(caption=f'Generator Output, communication round: {round_idx}')
                logging.info("########## Logging generator images... Complete #########")
                logging.info("########## Calculating FID Score...  #########")
                self.score_generator(round_idx)
                logging.info("########## Calculating FID Score... Complete #########")

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

    def log_gan_images(self, caption):
        generator = self.global_models.generator.to(self.device)
        generator.eval()
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

    def score_generator(self, round_idx):
        fake_size = 10000
        fake_data = self.global_models.generate_fake_dataset(fake_size, device=self.device,
                                                             batch_size=self.args.batch_size)
        fid_score = self.FIDScorer.calculate_fid(images_real=self.FID_source_set,
                                                 images_fake=fake_data, device=self.device)
        logging.info(f'FID Score: {fid_score}')
        wandb.log({'Gen/FID Score Distillation Set': fid_score, 'Round': round_idx})
