import logging

import wandb
from torchvision.utils import make_grid

from fedml_api.standalone.utils.BaseClient import BaseClient


class FedGDKDClient(BaseClient):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                 device, model_trainer):
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                         device,
                         model_trainer)

        # self.local_training_data, sizes = self.prepare_local_training_data(local_training_data)
        self.local_training_data = local_training_data
        self.distillation_dataset = None

        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        logging.info(f"""
            ########### Client {client_idx}: ###############
            Local sample number: {self.local_sample_number}
            Training dataset size: {self.get_dataset_size('train')}
            Test dataset size: {self.get_dataset_size('test')}
            ################################################
        """)

    def train(self, w_global, communication_round=0):
        logging.info(f'### Training Client {self.client_idx} ###')
        # self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        return self.model_trainer.get_model_params()

    def pre_train(self):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(private_data=self.local_training_data, device=self.device, args=self.args)

    def generate_distillation_set(self, noise_vector, class_labels):
        # Creating distillation dataset here to save memory but same as if sending noise vector to clients
        distillation_dataset = self.model_trainer.generate_distillation_dataset(noise_vector, class_labels,
                                                                                self.args.batch_size, self.device)
        self.distillation_dataset = distillation_dataset

    # def get_distillation_logits(self, w_global, noise_labels_loader):
    #     self.model_trainer.set_model_params(w_global)
    #     self.generate_distillation_set(noise_labels_loader)
    #     return self.model_trainer.get_classifier_logits(self.distillation_dataset, self.device)

    def get_distillation_logits(self, w_global, noise, class_labels, distillation_dataset=None):
        # self.model_trainer.set_model_params(w_global)
        self.generate_distillation_set(noise, class_labels)
        return self.model_trainer.get_classifier_logits(self.distillation_dataset, self.device)

    def classifier_knowledge_distillation(self, consensus_outputs, distillation_dataset=None):
        self.model_trainer.knowledge_distillation(self.distillation_dataset, consensus_outputs, self.device, self.args)

    def get_FID_score(self, fid_scorer, real_images, round_idx):
        logging.info(f"########## Calculating FID Score Client {self.client_idx}  #########")
        fid_score = fid_scorer.calculate_fid(images_real=real_images,
                                             images_fake=self.distillation_dataset, device=self.device)
        logging.info(f'FID Score: {fid_score}')
        wandb.log({f'Client {self.client_idx}/Gen/FID Score Distillation Set': fid_score, 'Round': round_idx})
        logging.info("########## Calculating FID Score Client {self.client_idx}... Complete #########")
        return fid_score

    def log_gan_images(self, caption, round_idx, fixed_noise, fixed_labels, denorm):
        generator = self.model_trainer.generator.to(self.device)
        generator.eval()
        images = make_grid(
            denorm(generator(fixed_noise.to(self.device), fixed_labels.to(self.device))),
            nrow=8,
            padding=2,
            normalize=False,
            range=None,
            scale_each=False, pad_value=0)
        images = wandb.Image(images, caption=caption)
        wandb.log({f"Client {self.client_idx}/Gen/Generator Outputs": images, 'Round': round_idx})
