import logging

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as tfs
from torchvision.utils import make_grid
import wandb
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle
import numpy as np

from fedml_api.model.cv.generator import Generator, ConditionalImageGenerator

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class ACGANModelTrainer(ModelTrainer):
    def __init__(self, generator, local_model, args):
        """
        Args:
            generator: Homogeneous model between clients that acts as knowledge transfer vehicle. In this case Generator
            local_model: Heterogeneous model that is chosen by clients that can better utilise client resources
        """
        super().__init__(generator)
        self.generator: ConditionalImageGenerator = generator
        self.local_model = local_model
        self.real_label, self.fake_label = 1, 0

        if args.client_optimizer == "sgd":
            self.optimiser_D = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)

        else:
            beta1, beta2 = 0.5, 0.999
            self.optimiser_D = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_model.parameters()),
                                                lr=args.lr,
                                                weight_decay=args.wd,
                                                amsgrad=True,
                                                betas=(beta1, beta2)
                                                )

    def get_model_params(self):
        return self.generator.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.generator.load_state_dict(model_parameters)

    def train(self, D_real, D_syn, fake_labels, device, args=None):
        """

        Args:
            train_data: Tuple of (labelled_data, unlabelled_data).
            device: Device to perform training on
            args: Other args
        Returns:

        """
        discriminator = self.local_model.to(device)

        # Initialize BCELoss function
        adversarial_loss = nn.BCELoss().to(device)
        auxiliary_loss = nn.CrossEntropyLoss().to(device)

        optimiser_D = self.optimiser_D

        batch_loss_D = []

        for (real_data, real_labels), (fake_data, fake_labels) in zip(D_real, D_syn):
            real_data, real_labels = real_data.to(device), real_labels.to(device)
            fake_data, fake_labels = fake_data.to(device), fake_labels.to(device)

            b_size = real_data.size(0)

            label_real_adv = torch.full((b_size, 1), self.real_label, device=device,
                                        requires_grad=False, dtype=torch.float)

            label_fake_adv = torch.full((b_size, 1), self.fake_label, device=device,
                                        requires_grad=False, dtype=torch.float)

            optimiser_D.zero_grad()

            # Loss for real images
            real_aux, real_pred = discriminator(real_data, discriminator=True)
            d_real_loss = (adversarial_loss(real_pred, label_real_adv) + auxiliary_loss(real_aux, real_labels)) / 2

            # Loss for fake images
            fake_aux, fake_pred = discriminator(fake_data.detach(), discriminator=True)
            d_fake_loss = (adversarial_loss(fake_pred, label_fake_adv) + auxiliary_loss(fake_aux, fake_labels)) / 2

            # Total discriminator loss
            errD = (d_real_loss + d_fake_loss) / 2
            errD.backward()
            optimiser_D.step()

            batch_loss_D.append(errD.item())

        return sum(batch_loss_D) / len(batch_loss_D)

    def _discriminator_output(self, logits):
        Z_x = torch.logsumexp(logits, dim=-1)
        return torch.sigmoid(Z_x / (Z_x + 1))

    def _gan_training(self, generator: ConditionalImageGenerator, discriminator, labelled_data, unlabelled_data, epochs,
                      optimizer_G, optimizer_D, device):
        generator.train()
        discriminator.train()
        real_label, fake_label = 1, 0  # Soft labels

        # train_data = labelled_data if unlabelled_data is None else zip(labelled_data, unlabelled_data)

        # Initialize BCELoss function
        adversarial_loss = nn.BCELoss().to(device)
        auxiliary_loss = nn.CrossEntropyLoss().to(device)

        torch.autograd.set_detect_anomaly(True)

        epoch_loss_D = []
        epoch_loss_G = []
        batch_acc_D = []
        for epoch in range(epochs):
            batch_loss_D, batch_loss_G = [], []
            # train_data = labelled_data if unlabelled_data is None else zip(labelled_data, unlabelled_data)
            for batch_idx, data in enumerate(
                    labelled_data if unlabelled_data is None else zip(labelled_data, cycle(unlabelled_data))):
                if unlabelled_data is not None:
                    (real, labels), (synth, synth_labels) = data
                    # ulreal = ulreal[0]  # zip packs iterables of varying length in to tuples so ulreal becomes [
                    # ulreal]
                    real, labels, synth, synth_labels = real.to(device), labels.to(device), synth.to(
                        device), synth_labels.to(device)
                    real = torch.unsqueeze(real, 1) if len(
                        real.shape) < 4 else real  # 1 channel datasets miss second dim
                    with torch.no_grad():
                        real, labels = torch.cat((real, synth), dim=0), torch.cat((labels, synth_labels), dim=0)
                else:
                    real, labels = data
                    real, labels = real.to(device), labels.to(device)
                    # unlabelled_real = real

                # unlabelled_real = unlabelled_real.to(device)
                b_size = real.size(0)

                label_real_adv = torch.full((b_size, 1), real_label, device=device,
                                            requires_grad=False, dtype=torch.float)
                label_fake_adv = torch.full((b_size, 1), fake_label, device=device,
                                            requires_grad=False, dtype=torch.float)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = generator.generate_noise_vector(b_size, device=device)
                gen_labels = generator.generate_random_labels(b_size, device=device)

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                pred_label, validity = discriminator(gen_imgs, discriminator=True)
                errG = (adversarial_loss(validity, label_real_adv) + auxiliary_loss(pred_label, gen_labels)) / 2

                errG.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Loss for real images
                real_aux, real_pred = discriminator(real, discriminator=True)
                d_real_loss = (adversarial_loss(real_pred, label_real_adv) + auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_aux, fake_pred = discriminator(gen_imgs.detach(), discriminator=True)
                d_fake_loss = (adversarial_loss(fake_pred, label_fake_adv) + auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                errD = (d_real_loss + d_fake_loss) / 2
                errD.backward()
                optimizer_D.step()

                # Save Losses for plotting later
                batch_loss_G.append(errD.item())
                batch_loss_D.append(errG.item())

                # Logging
                epoch_loss_G.append(sum(batch_loss_G) / len(batch_loss_G))
                epoch_loss_D.append(sum(batch_loss_D) / len(batch_loss_D))

            logging.info(
                f'tEpoch: {epoch}\t Gen Loss: {sum(epoch_loss_G) / len(epoch_loss_D):.6f}\t Disc Loss: {sum(epoch_loss_D) / len(epoch_loss_D)}')

    def _train_loop(self, model, train_data, criterion, epochs, optimizer, device):
        """

        Args:
            model: Model to be trained
            train_data: Training data
            criterion: Loss for main task
            epochs: Epochs of training to be completed
            optimizer: Optimiser that should be used to update hyperparams
            device: Device in which training should occur
        Returns:

        """

        model.train()
        transforms = self.transforms.to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        epoch_loss = []
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                x = transforms(x)

                model.zero_grad()
                output = model(x)  # in classification case will be logits
                # 3. on labeled data
                logz_label = torch.logsumexp(output, dim=-1)
                prob_label = torch.gather(output, 1, labels.unsqueeze(1))
                loss = -torch.mean(prob_label) + torch.mean(logz_label)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(f'tEpoch: {epoch}\tLoss: {sum(epoch_loss) / len(epoch_loss):.6f}')
        return epoch_loss

    def test(self, test_data, device, args=None):
        return self.test_model(self.local_model, test_data, device, args)

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def pre_train(self, private_data, device, args):
        """
               Pre-training in FedMD algorithm to do transfer learning from public data set
               to private dataset

               Args:
                   private_data: Private data only known to the client
                   device: Device to perform training on
                   args: Other args
               Returns:

               """
        model = self.local_model
        model.to(device)

        if args.client_optimizer == "sgd":
            optimiser_D = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)

        else:
            beta1, beta2 = 0.5, 0.999
            optimiser_D = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_model.parameters()),
                                           lr=args.lr,
                                           weight_decay=args.wd,
                                           amsgrad=True,
                                           betas=(beta1, beta2)
                                           )

        # Transfer learning to private dataset
        self._train_loop(model, train_data=private_data, criterion=None, epochs=args.pretrain_epochs_private,
                         optimizer=optimiser_D, device=device)

    def _get_pseudo_labels_with_probability(self, disc_logits):
        class_probabilities = F.softmax(disc_logits, dim=-1)
        max_probs, labels = class_probabilities.max(dim=-1)
        return max_probs, labels

    def generate_synthetic_dataset(self, target_size, real_score_threshold=0.85, device='cpu'):
        generator, discriminator = self.generator.to(device), self.local_model.to(device)
        generator.eval()
        discriminator.eval()

        labels = generator.generate_balanced_labels(target_size, device=device)
        noise = generator.generate_noise_vector(target_size, device=device)
        generated_images = generator(noise, labels)

        # Filter by realness score to select best generated images
        label_probs, realness = discriminator(generated_images, discriminator=True)
        mask = realness >= real_score_threshold
        good_generated_images, labels = generated_images[mask], labels[mask]

        # If generator is not good enough do not create synthetic dataset
        size = good_generated_images.size(0)
        if size == 0:
            return None, 0

        dataset = TensorDataset(good_generated_images, labels)
        # data_loader = DataLoader(dataset, batch_size=batch_size)
        return dataset, size

    def get_discriminator_output(self, D_syn, device):
        discriminator = self.local_model.to(device)
        discriminator.eval()

        gan_out_all, cls_logits_all = [], []
        for data, _ in D_syn:
            cls_logits, gan_out = discriminator(data, discriminator=True)
            gan_out_all.append(gan_out)
            cls_logits_all.append(cls_logits)

        return torch.cat(gan_out_all, 0), torch.cat(cls_logits_all, 0)
