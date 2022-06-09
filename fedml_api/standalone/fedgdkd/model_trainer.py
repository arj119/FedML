import logging

import torch
from torch import nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from fedml_api.model.cv.generator import Generator, ConditionalImageGenerator
from fedml_api.standalone.fedgdkd.ac_gan_model_trainer import ACGANModelTrainer
from knowledge_distillation.soft_target import SoftTarget

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedGDKDModelTrainer(ACGANModelTrainer):
    def _gan_training(self, generator: ConditionalImageGenerator, discriminator, train_data, epochs, optimiser_G,
                      optimiser_D, device):
        generator.train()
        discriminator.train()
        torch.autograd.set_detect_anomaly(True)
        auxiliary_loss = nn.CrossEntropyLoss().to(device)

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

                # 2. on the generated data
                cls_logits_fake = discriminator(gen_imgs.detach())  # detach() because we are not training G here
                logz_fake = torch.logsumexp(cls_logits_fake, dim=-1)
                prob_label_fake = torch.gather(cls_logits_fake, 1, gen_labels.unsqueeze(-1))
                aux_loss_fake = -torch.mean(prob_label_fake) + torch.mean(logz_fake)
                adv_loss_fake = torch.mean(F.softplus(logz_fake))
                d_fake_loss = 0.5 * (aux_loss_fake + adv_loss_fake)

                # 3. on labeled data
                # 1. on Unlabelled data
                cls_logits_real = discriminator(real)
                logz_real = torch.logsumexp(cls_logits_real, dim=-1)
                prob_label_real = torch.gather(cls_logits_real, 1, labels.unsqueeze(-1))
                aux_loss_real = -torch.mean(prob_label_real) + torch.mean(logz_real)
                adv_loss_real = -torch.mean(logz_real) + torch.mean(F.softplus(logz_real))
                d_real_loss = 0.5 * (aux_loss_real + adv_loss_real)

                errD = d_real_loss + d_fake_loss

                # Loss for real images
                # cls_logits_real = discriminator(real)
                # gan_logits_real = torch.logsumexp(cls_logits_real, dim=-1)
                # d_real_loss = (auxiliary_loss(cls_logits_real, labels) + torch.mean(gan_logits_real)) / 2
                #
                # Loss for fake images
                # cls_logits_fake = discriminator(gen_imgs.detach())  # detach() because we are not training G here
                # gan_logits_fake = torch.logsumexp(cls_logits_fake, dim=-1)
                # d_fake_loss = (torch.mean(F.softplus(gan_logits_fake)) + auxiliary_loss(cls_logits_fake,
                # gen_labels)) / 2

                # Total discriminator loss
                # errD = (d_real_loss + d_fake_loss) / 2
                errD.backward()
                optimiser_D.step()

                # Save Losses for plotting later
                batch_loss_G.append(errD.item())
                batch_loss_D.append(errG.item())

                # Logging
                epoch_loss_G.append(sum(batch_loss_G) / len(batch_loss_G))
                epoch_loss_D.append(sum(batch_loss_D) / len(batch_loss_D))

            logging.info(
                f'tEpoch: {epoch}\t Gen Loss: {epoch_loss_G[-1]:.6f}\t Disc Loss: {epoch_loss_D[-1]:.6f}')

        return adv_loss_fake.item(), aux_loss_fake.item(), d_fake_loss.item() * 2



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
