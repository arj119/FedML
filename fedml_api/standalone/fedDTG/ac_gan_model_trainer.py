import copy
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
from knowledge_distillation.soft_target import SoftTarget

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class ACGANModelTrainer(ModelTrainer):
    def __init__(self, generator, discriminator, local_model):
        """
        Args:
            generator: Homogeneous model between clients that acts as knowledge transfer vehicle. In this case Generator
            local_model: Heterogeneous model that is chosen by clients that can better utilise client resources
        """
        super().__init__(generator)
        self.generator: ConditionalImageGenerator = generator
        self.discriminator = discriminator
        self.local_model = local_model

    def get_model_params(self):
        return self.generator.cpu().state_dict(), self.discriminator.cpu().state_dict()

    def set_model_params(self, model_parameters):
        g, d = model_parameters
        self.generator.load_state_dict(g)
        self.discriminator.load_state_dict(d)

    def train(self, train_data, device, args=None):
        """

        Args:
            train_data: Tuple of (labelled_data, unlabelled_data).
            device: Device to perform training on
            args: Other args
        Returns:

        """
        generator, discriminator, local_model = self.generator.to(device), self.discriminator.to(
            device), self.local_model.to(device)

        optimiser_G = self.get_client_optimiser(generator, args.gen_optimizer, args.gen_lr)
        optimiser_D = self.get_client_optimiser(discriminator, args.client_optimizer, args.lr)
        optimiser_C = self.get_client_optimiser(local_model, args.client_optimizer, args.lr)

        self._gan_training(generator, discriminator, local_model, train_data, args.epochs, optimiser_G,
                           optimiser_D, optimiser_C, device)

    def _gan_training(self, generator: ConditionalImageGenerator, discriminator, classifier, train_data, epochs,
                      optimiser_G, optimiser_D, optimiser_C, device):
        generator.train()
        discriminator.train()
        classifier.train()
        real_label, fake_label = 0.9, 0  # Soft labels

        # Initialize BCELoss function
        adversarial_loss = nn.BCELoss().to(device)
        auxiliary_loss = nn.CrossEntropyLoss().to(device)

        torch.autograd.set_detect_anomaly(True)

        epoch_loss_D = []
        epoch_loss_G = []
        epoch_loss_C = []
        for epoch in range(epochs):
            batch_loss_D, batch_loss_G, batch_loss_C = [], [], []
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
                _, validity = discriminator(gen_imgs, discriminator=True)
                pred_label = classifier(gen_imgs)
                gradient_reversal = pred_label.register_hook(lambda grad: -grad)  # gradient reversal
                errG = (adversarial_loss(validity, label_real_adv) + auxiliary_loss(pred_label, gen_labels)) / 2
                gradient_reversal.remove()  # remove gradient reversal hook

                errG.backward()
                optimiser_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimiser_D.zero_grad()

                # Loss for real images
                _, real_pred = discriminator(real, discriminator=True)
                d_real_loss = adversarial_loss(real_pred, label_real_adv)

                # Loss for fake images
                fake_aux, fake_pred = discriminator(gen_imgs.detach(), discriminator=True)
                d_fake_loss = adversarial_loss(fake_pred, label_fake_adv)

                # Total discriminator loss
                errD = (d_real_loss + d_fake_loss) / 2
                errD.backward()
                optimiser_D.step()

                # ---------------------
                #  Train Classifier
                # ---------------------

                optimiser_C.zero_grad()

                # Loss for real images
                pred_label_real = classifier(real)
                c_real_loss = auxiliary_loss(pred_label_real, labels)

                pred_label_fake = classifier(gen_imgs.detach())
                c_fake_loss = auxiliary_loss(pred_label_fake, gen_labels)

                errC = (c_real_loss + c_fake_loss) / 2
                errC.backward()
                optimiser_C.step()

                # Save Losses for plotting later
                batch_loss_G.append(errD.item())
                batch_loss_D.append(errG.item())
                batch_loss_C.append(errC.item())

            # Logging
            epoch_loss_G.append(sum(batch_loss_G) / len(batch_loss_G))
            epoch_loss_D.append(sum(batch_loss_D) / len(batch_loss_D))
            epoch_loss_C.append(sum(batch_loss_C) / len(batch_loss_C))

            logging.info(
                f'tEpoch: {epoch}\t Gen Loss: {epoch_loss_G[-1]:.6f}\t Disc Loss: {epoch_loss_D[-1]:.6f}\t Classifier Loss: {epoch_loss_C[-1]:.6f}')

    def get_classifier_logits(self, distillation_dataset, device):
        """

        Args:
            noise: A noise tensor of shape (b_size,)
            class_labels: A list of all possible classes to iterate through
            device: Device to perform computation on either 'cpu' or 'gpu:...'

        Returns:
            A concatenation of classifier logits for each sample

        """
        generator, classifier = self.generator.to(device), self.local_model.to(device)
        generator.eval()
        classifier.eval()

        with torch.no_grad():
            logits = []
            for synth_data, _ in distillation_dataset:
                synth_data = synth_data.to(device)
                logits.append(classifier(synth_data))
        return torch.cat(logits).detach().cpu()

    def knowledge_distillation(self, distillation_dataset: DataLoader, consensus_logits: DataLoader, device, args):
        assert len(distillation_dataset) == len(
            consensus_logits), f"distillation_dataset of size {len(distillation_dataset)}, vs consensus logits of size {len(consensus_logits)}"

        classifier = self.local_model.to(device)
        classifier.train()

        kd_criterion = SoftTarget(T=4).to(device)
        cls_criterion = nn.CrossEntropyLoss().to(device)

        kd_alpha = args.kd_alpha

        optimiser_C = self.get_client_optimiser(classifier, args.client_optimizer, args.lr)

        epoch_dist_loss = []
        for epoch in range(args.kd_epochs):
            batch_dist_loss = []
            for idx, data in enumerate(zip(distillation_dataset, consensus_logits)):
                (synth_data, labels), consensus_logits_batch = data
                consensus_logits_batch = consensus_logits_batch[0]  # As tensor dataset wraps in a list
                synth_data, labels, consensus_logits_batch = synth_data.to(device), labels.to(
                    device), consensus_logits_batch.to(device)

                optimiser_C.zero_grad()

                cls_logits = classifier(synth_data)

                diss_loss = (1 - kd_alpha) * cls_criterion(cls_logits, labels) + kd_alpha * kd_criterion(cls_logits,
                                                                                                         consensus_logits_batch)
                diss_loss.backward()
                optimiser_C.step()

                batch_dist_loss.append(diss_loss.item())

            epoch_dist_loss.append(sum(batch_dist_loss) / len(batch_dist_loss))
            logging.info(
                f'tEpoch: {epoch}\t Diss loss {epoch_dist_loss[-1]:.6f}')

    def test(self, test_data, device, args=None):
        model = self.local_model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        if args.dataset == "stackoverflow_lr":
            criterion = nn.BCELoss(reduction='sum').to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                if args.dataset == "stackoverflow_lr":
                    predicted = (pred > .5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics['test_precision'] += precision.sum().item()
                    metrics['test_recall'] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics['test_total'] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics['test_total'] += target.size(0) * target.size(1)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def _get_pseudo_labels_with_probability(self, disc_logits):
        class_probabilities = F.softmax(disc_logits, dim=-1)
        max_probs, labels = class_probabilities.max(dim=-1)
        return max_probs, labels

    def generate_distillation_dataset(self, noise_labels: DataLoader, device):
        generator = self.generator.to(device)
        generator.eval()
        with torch.no_grad():
            synth_data = []
            # labels_bucket = []
            for noise, labels in noise_labels:
                noise, labels = noise.to(device), labels.to(device)
                generated_data = generator(noise, labels)
                synth_data.append(generated_data.cpu())
                # labels_bucket.append(copy.deepcopy(labels.cpu()))

            synth_data = torch.cat(synth_data, dim=0)
        # labels = torch.cat(labels_bucket, dim=0)

        # dataset = TensorDataset(synth_data, labels)
        # data_loader = DataLoader(dataset, batch_size=noise_labels.batch_size)
        return synth_data
