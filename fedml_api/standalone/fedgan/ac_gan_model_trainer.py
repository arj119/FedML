import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedml_api.model.cv.generator import ConditionalImageGenerator

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class ACGANModelTrainer(ModelTrainer):
    def __init__(self, generator, disc_classifier):
        """
        Args:
            generator: Homogeneous model between clients that acts as knowledge transfer vehicle. In this case Generator
            local_model: Heterogeneous model that is chosen by clients that can better utilise client resources
        """
        super().__init__(generator)
        self.generator: ConditionalImageGenerator = generator
        self.disc_classifier = disc_classifier

    def get_model_params(self):
        return self.generator.cpu().state_dict(), self.disc_classifier.cpu().state_dict()

    def set_model_params(self, model_parameters):
        gen, disc = model_parameters
        self.generator.load_state_dict(gen)
        self.disc_classifier.load_state_dict(disc)

    def train(self, train_data, device, args=None):
        """

        Args:
            train_data: Tuple of (labelled_data, unlabelled_data).
            device: Device to perform training on
            args: Other args
        Returns:

        """
        generator, discriminator = self.generator.to(device), self.disc_classifier.to(device)

        optimiser_G = self.get_client_optimiser(generator, args.gen_optimizer, args.gen_lr)
        optimiser_D = self.get_client_optimiser(discriminator, args.client_optimizer, args.lr)

        self._gan_training(generator, discriminator, train_data, args.epochs, optimiser_G, optimiser_D, device)

    def _gan_training(self, generator: ConditionalImageGenerator, discriminator, train_data, epochs, optimiser_G,
                      optimiser_D, device):
        generator.train()
        discriminator.train()
        real_label, fake_label = 0.9, 0  # Soft labels

        # Initialize BCELoss function
        adversarial_loss = nn.BCELoss().to(device)
        auxiliary_loss = nn.CrossEntropyLoss().to(device)

        epoch_loss_D = []
        epoch_loss_G = []
        for epoch in range(epochs):
            batch_loss_D, batch_loss_G = [], []
            # train_data = labelled_data if unlabelled_data is None else zip(labelled_data, unlabelled_data)
            for batch_idx, (real, labels) in enumerate(train_data):
                real, labels = real.to(device), labels.to(device)

                b_size = real.size(0)

                label_real_adv = torch.full((b_size, 1), real_label, device=device,
                                            requires_grad=False, dtype=torch.float)
                label_fake_adv = torch.full((b_size, 1), fake_label, device=device,
                                            requires_grad=False, dtype=torch.float)

                # -----------------
                #  Train Generator
                # -----------------

                optimiser_G.zero_grad()

                # Sample noise and labels as generator input
                z = generator.generate_noise_vector(b_size, device=device)
                gen_labels = generator.generate_random_labels(b_size, device=device)

                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                pred_label, validity = discriminator(gen_imgs, discriminator=True)
                errG = (adversarial_loss(validity, label_real_adv) + auxiliary_loss(pred_label, gen_labels)) / 2

                errG.backward()
                optimiser_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimiser_D.zero_grad()

                # Loss for real images
                real_aux, real_pred = discriminator(real, discriminator=True)
                d_real_loss = (adversarial_loss(real_pred, label_real_adv) + auxiliary_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_aux, fake_pred = discriminator(gen_imgs.detach(), discriminator=True)
                d_fake_loss = (adversarial_loss(fake_pred, label_fake_adv) + auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                errD = (d_real_loss + d_fake_loss) / 2
                errD.backward()
                optimiser_D.step()

                # ---------------------
                #  Train Classifier
                # ---------------------

                # Save Losses for plotting later
                batch_loss_G.append(errD.item())
                batch_loss_D.append(errG.item())

            # Logging
            epoch_loss_G.append(sum(batch_loss_G) / len(batch_loss_G))
            epoch_loss_D.append(sum(batch_loss_D) / len(batch_loss_D))

            logging.info(
                f'tEpoch: {epoch}\t Gen Loss: {epoch_loss_G[-1]:.6f}\t Disc Loss: {epoch_loss_D[-1]:.6f}')

    def test(self, test_data, device, args=None):
        return self.test_model(self.disc_classifier, test_data, device, args)

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def generate_fake_dataset(self, size, device, batch_size):
        generator = self.generator.to(device)
        generator.eval()

        noise_vector = generator.generate_noise_vector(size, device=device)
        labels_vector = generator.generate_balanced_labels(size, device=device)

        noise_labels = TensorDataset(noise_vector, labels_vector)
        noise_labels_loader = DataLoader(noise_labels, batch_size=batch_size)

        with torch.no_grad():
            synth_data = []
            for noise, labels in noise_labels_loader:
                noise, labels = noise.to(device), labels.to(device)
                generated_data = generator(noise, labels)
                synth_data.append(generated_data.cpu())
            synth_data = torch.cat(synth_data, dim=0)

        assert synth_data.size(0) == labels_vector.size(0), f'{synth_data.size(0)} != {labels_vector.size(0)}'
        fake_data = DataLoader(TensorDataset(synth_data, labels_vector), batch_size=batch_size)

        del noise_labels_loader
        return fake_data
