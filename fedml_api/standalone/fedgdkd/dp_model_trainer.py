import logging

import torch
from opacus.optimizers import DPOptimizer
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fedml_api.model.cv.generator import ConditionalImageGenerator
from fedml_api.standalone.fedgdkd.utils.custom_privacy_engine import CustomPrivacyEngine
from knowledge_distillation.soft_target import SoftTarget
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class DPModelTrainer(ModelTrainer):
    def __init__(self, generator, local_model):
        """
        Args:
            generator: Homogeneous model between clients that acts as knowledge transfer vehicle. In this case Generator
            local_model: Heterogeneous model that is chosen by clients that can better utilise client resources
        """
        super().__init__(generator)

        # Make generator DP compatible
        self.generator: ConditionalImageGenerator = generator
        if local_model is not None:
            errors = ModuleValidator.validate(local_model)
            logging.info(errors)
        self.local_model = ModuleValidator.fix_and_validate(local_model) if local_model is not None else None

        # Privacy Engine
        self.privacy_engine = PrivacyEngine()

    def get_model_params(self):
        return self.generator.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.generator.load_state_dict(model_parameters)

    def train(self, train_data, device, args=None):
        """

        Args:
            train_data: Tuple of (labelled_data, unlabelled_data).
            device: Device to perform training on
            args: Other args
        Returns:

        """

        len_training_data = self.get_dataset_size(train_data)

        DELTA = 1 / len_training_data
        EPSILON = 50.0
        MAX_GRAD_NORM = 1.2

        generator, discriminator = self.generator.to(device), self.local_model.to(device)

        optimiser_G = self.get_client_optimiser(generator, args.gen_optimizer, args.gen_lr)
        optimiser_D = self.get_client_optimiser(discriminator, args.client_optimizer, args.lr)

        discriminator, optimiser_D, train_data = self.privacy_engine.make_private_with_epsilon(
            module=discriminator,
            optimizer=optimiser_D,
            data_loader=train_data,
            epochs=args.epochs,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )

        self._gan_training(generator, discriminator, train_data, args.epochs, optimiser_G, optimiser_D, DELTA, device)

    def _gan_training(self, generator: ConditionalImageGenerator, discriminator, train_data, epochs, optimiser_G,
                      optimiser_D, delta, device):
        generator.train()
        discriminator.train()

        epoch_loss_D = []
        epoch_loss_G = []
        for epoch in range(epochs):
            batch_loss_D, batch_loss_G = [], []
            # train_data = labelled_data if unlabelled_data is None else zip(labelled_data, unlabelled_data)
            for batch_idx, (real, labels) in enumerate(train_data):
                real, labels = real.to(device), labels.to(device)

                b_size = real.size(0)

                # -----------------
                #  Train Generator
                # -----------------

                optimiser_G.zero_grad()

                # Sample noise and labels as generator input
                z = generator.generate_noise_vector(b_size, device=device)
                gen_labels = generator.generate_random_labels(b_size, device=device)

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                cls_logits_gen = discriminator(gen_imgs)
                logz_gen = torch.logsumexp(cls_logits_gen, dim=-1)
                prob_label = torch.gather(cls_logits_gen, 1, gen_labels.unsqueeze(-1))
                aux_loss = -torch.mean(prob_label) + torch.mean(logz_gen)
                adv_loss = -torch.mean(logz_gen) + torch.mean(F.softplus(logz_gen))
                errG = (adv_loss + aux_loss) / 2
                errG.backward()
                optimiser_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # TRAIN THE DISCRIMINATOR (THE CLASSIFIER)
                optimiser_D.zero_grad()
                #
                # # 2. on the generated data
                # cls_logits_fake = discriminator(gen_imgs.detach())  # detach() because we are not training G here
                # logz_fake = torch.logsumexp(cls_logits_fake, dim=-1)
                # prob_label_fake = torch.gather(cls_logits_fake, 1, gen_labels.unsqueeze(-1))
                # aux_loss_fake = -torch.mean(prob_label_fake) + torch.mean(logz_fake)
                # adv_loss_fake = torch.mean(F.softplus(logz_fake))
                # d_fake_loss = 0.5 * (aux_loss_fake + adv_loss_fake)

                # 3. on labeled data
                # 1. on Unlabelled data
                cls_logits_real = discriminator(real)
                logz_real = torch.logsumexp(cls_logits_real, dim=-1)
                prob_label_real = torch.gather(cls_logits_real, 1, labels.unsqueeze(-1))
                aux_loss_real = -torch.mean(prob_label_real) + torch.mean(logz_real)
                adv_loss_real = -torch.mean(logz_real) + torch.mean(F.softplus(logz_real))
                d_real_loss = 0.5 * (aux_loss_real + adv_loss_real)

                errD = d_real_loss #+ d_fake_loss

                errD.backward()
                optimiser_D.step()

                # Save Losses for plotting later
                batch_loss_G.append(errD.item())
                batch_loss_D.append(errG.item())

                # Logging
                epoch_loss_G.append(sum(batch_loss_G) / len(batch_loss_G))
                epoch_loss_D.append(sum(batch_loss_D) / len(batch_loss_D))

            epsilon = self.privacy_engine.get_epsilon(delta)
            logging.info(
                f'tEpoch: {epoch}\t Gen Loss: {epoch_loss_G[-1]:.6f}\t Disc Loss: {epoch_loss_D[-1]:.6f}\t epsilon={epsilon}\t delta={delta}')

    def get_classifier_logits(self, distillation_dataset, device):
        """

        Args:
            noise: A noise tensor of shape (b_size,)
            class_labels: A list of all possible classes to iterate through
            device: Device to perform computation on either 'cpu' or 'gpu:...'

        Returns:
            A concatenation of classifier logits for each sample

        """
        classifier = self.local_model.to(device)
        classifier.eval()

        with torch.no_grad():
            logits = []
            for synth_data, _ in distillation_dataset:
                synth_data = synth_data.to(device)
                pred_logits, validity = classifier(synth_data, discriminator=True)
                logits.append(pred_logits)
        return torch.cat(logits).detach().cpu()

    def knowledge_distillation(self, distillation_dataset: DataLoader, consensus_outputs: DataLoader, device, args):
        assert len(distillation_dataset) == len(
            consensus_outputs), f"distillation_dataset of size {len(distillation_dataset)}, vs consensus logits of size {len(consensus_outputs)}"

        classifier = self.local_model.to(device)
        classifier.train()

        kd_criterion_logits = SoftTarget(T=4).to(device)
        cls_criterion = nn.CrossEntropyLoss().to(device)

        kd_alpha = args.kd_alpha

        optimiser_D = self.get_client_optimiser(classifier, args.client_optimizer, args.lr)

        epoch_dist_loss, epoch_kd_loss = [], []
        for epoch in range(args.kd_epochs):
            batch_dist_loss, batch_kd_loss = [], []
            for idx, data in enumerate(zip(distillation_dataset, consensus_outputs)):
                (synth_data, labels), t_logits = data
                t_logits = t_logits[0]
                synth_data, labels, t_logits = synth_data.to(device), labels.to(device), t_logits.to(
                    device)

                optimiser_D.zero_grad()

                cls_logits = classifier(synth_data)

                kd_loss = kd_criterion_logits(cls_logits, t_logits)
                diss_loss = (1 - kd_alpha) * cls_criterion(cls_logits, labels) + kd_alpha * kd_loss
                diss_loss.backward()
                optimiser_D.step()

                batch_dist_loss.append(diss_loss.item())
                batch_kd_loss.append(kd_loss.item())

            epoch_dist_loss.append(sum(batch_dist_loss) / len(batch_dist_loss))
            epoch_kd_loss.append(sum(batch_kd_loss) / len(batch_kd_loss))

            logging.info(
                f'tEpoch: {epoch}\t Diss loss {epoch_dist_loss[-1]:.6f}\t KD loss {epoch_kd_loss[-1]:.6f}')

    def test(self, test_data, device, args=None):
        return self.test_model(self.local_model, test_data, device, args)

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def generate_distillation_dataset(self, noise_labels: DataLoader, device):
        generator = self.generator.to(device)
        generator.eval()
        with torch.no_grad():
            synth_data = []
            for noise, labels in noise_labels:
                noise, labels = noise.to(device), labels.to(device)
                generated_data = generator(noise, labels)
                synth_data.append(generated_data.cpu())
            synth_data = torch.cat(synth_data, dim=0)

        return synth_data

    def get_dataset_size(self, ds):
        if isinstance(ds, torch.utils.data.DataLoader):
            size = len(ds.dataset)
        else:
            size = len(ds)
        return size
