import copy
import logging
from typing import List, Tuple

import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
import torchvision.transforms as tfs

from FID.FIDScorer import FIDScorer
from fedml_api.standalone.fedgdkd.ac_gan_model_trainer import ACGANModelTrainer
from fedml_api.standalone.fedgdkd.client import FedGDKDClient
from fedml_api.standalone.fedgdkd.model_trainer import FedGDKDModelTrainer
from fedml_api.standalone.utils.HeterogeneousModelBaseTrainerAPI import HeterogeneousModelBaseTrainerAPI


class FedGDKDAPI(HeterogeneousModelBaseTrainerAPI):
    def __init__(self, dataset, device, args, generator, client_models: List[Tuple[torch.nn.Module, int]]):
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

        self.generator = ACGANModelTrainer(generator, None)
        logging.info(generator)
        self.generator_model = self.generator.generator
        # For logging GAN progress
        self.fixed_labels = self.generator_model.generate_balanced_labels(
            self.generator_model.num_classes * 8,
            device='cpu')
        self.fixed_noise = self.generator_model.generate_noise_vector(self.generator_model.num_classes * 8,
                                                                      device='cpu')

        self._setup_clients(self.train_data_local_num_dict, self.train_data_local_dict, self.test_data_local_dict,
                            client_models)

        self._plot_client_training_data_distribution()

        # Generate dataset that can be used to calculate FID score
        self.FID_source_set = self._generate_train_subset(num_samples=10000)
        self.FIDScorer = FIDScorer()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
                       client_models):
        logging.info("############setup_clients (START)#############")

        c_idx = 0
        for local_model, freq in client_models:
            for i in range(freq):
                model_trainer = FedGDKDModelTrainer(
                    copy.deepcopy(self.generator.model),
                    copy.deepcopy(local_model)
                )
                c = FedGDKDClient(c_idx, train_data_local_dict[c_idx], test_data_local_dict[c_idx],
                                  train_data_local_num_dict[c_idx], self.test_global, self.args, self.device,
                                  model_trainer)
                c_idx += 1
                self.client_list.append(c)

        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.generator.get_model_params()
        DISTILLATION_DATASET_SIZE = self.args.distillation_dataset_size
        distillation_dataset = None
        teacher_logits = None
        prev_client_subset = None

        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            client_subset = self._client_sampling(round_idx)

            # ---------------
            # GAN Training
            # ---------------
            logging.info('########## Gan Training ########')

            w_locals = []

            client: FedGDKDClient
            for client in client_subset:
                # # Perform knowledge distillation (model drift correction) on current participating clients
                # if prev_client_subset is not None and client.client_idx not in prev_client_subset:
                #     logging.info(f"######## KD for new client {client.client_idx} ########")
                #     assert distillation_dataset is not None and teacher_logits is not None, 'Need both to perform KD'
                #     # Calculate teacher logits as mean of logits belonging to other clients
                #     client.classifier_knowledge_distillation(teacher_logits, distillation_dataset)

                # Perform local training as usual
                w_local = client.train(copy.deepcopy(w_global), round_idx)

                w_locals.append(w_local)

            # update global weights
            w_global = self._aggregate(w_locals)
            # self.generator.set_model_params(g_global)
            # self.discriminator.set_model_params(d_global)
            self.generator.set_model_params(w_global)

            # ---------------
            # Distillation Phase
            # ---------------
            # logging.info('########## Distillation ########')

            # Creating distillation dataset here to save memory but same as if sending noise vector to clients
            distillation_dataset = self.generate_fake_dataset(DISTILLATION_DATASET_SIZE)

            # local_logits = []
            # logging.info("########## Acquiring distillation logits... #########")
            # for client in client_subset:
            #     logits = client.get_distillation_logits(copy.deepcopy(w_global), distillation_dataset)
            #     local_logits.append(logits)
            #     logging.info(f"Client {client.client_idx} complete")
            #
            # # Calculate average soft labels
            # logging.info(f"######## Knowledge distillation stage ########")
            # for idx, client in enumerate(client_subset):
            #     # Calculate teacher logits for client
            #     logging.info(f"##### Client {client.client_idx} #####")
            #     teacher_logits = torch.mean(torch.stack(local_logits[:idx] + local_logits[idx + 1:]), dim=0)
            #     teacher_logits = DataLoader(TensorDataset(teacher_logits), batch_size=self.args.batch_size)
            #
            #     client.classifier_knowledge_distillation(teacher_logits, distillation_dataset)

            # For next round
            # teacher_logits = torch.mean(torch.stack(local_logits), dim=0)
            # teacher_logits = DataLoader(TensorDataset(teacher_logits), batch_size=self.args.batch_size)
            # prev_client_subset = {c.client_idx for c in client_subset}

            if round_idx % 1 == 0:
                logging.info("########## Logging generator images... #########")
                self.log_gan_images(caption=f'Generator Output, communication round: {round_idx}', round_idx=round_idx)
                logging.info("########## Logging generator images... Complete #########")
                logging.info("########## Calculating FID Score...  #########")
                fake = distillation_dataset
                if DISTILLATION_DATASET_SIZE != 10000:
                    fake = self.generate_fake_dataset(DISTILLATION_DATASET_SIZE)
                fid_score = self.FIDScorer.calculate_fid(images_real=self.FID_source_set,
                                                         images_fake=fake, device=self.device)
                if DISTILLATION_DATASET_SIZE != 10000:
                    del fake
                logging.info(f'FID Score: {fid_score}')
                wandb.log({'Gen/FID Score Distillation Set': fid_score, 'Round': round_idx})
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

    def log_gan_images(self, caption, round_idx):
        images = make_grid(
            self.denorm(
                self.generator_model(self.fixed_noise.to(self.device), self.fixed_labels.to(self.device))),
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

    def generate_fake_dataset(self, size):
        # Creating distillation dataset here to save memory but same as if sending noise vector to clients
        noise_vector = self.generator_model.generate_noise_vector(size, device=self.device)
        labels = self.generator_model.generate_balanced_labels(size, device=self.device)
        noise_labels = TensorDataset(noise_vector, labels)
        noise_labels_loader = DataLoader(noise_labels, batch_size=self.args.batch_size)

        synth_data = self.generator.generate_distillation_dataset(noise_labels_loader, device=self.device)
        del noise_labels_loader
        return DataLoader(TensorDataset(synth_data, labels), batch_size=self.args.batch_size)
