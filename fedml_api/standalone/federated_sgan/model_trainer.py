import logging

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as tfs
from torchvision.utils import make_grid
import wandb
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle

from fedml_api.model.cv.generator import Generator

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedSSGANModelTrainer(ModelTrainer):
    def __init__(self, generator, local_model):
        """
        Args:
            generator: Homogeneous model between clients that acts as knowledge transfer vehicle. In this case Generator
            local_model: Heterogeneous model that is chosen by clients that can better utilise client resources
        """
        super().__init__(generator)
        self.generator: Generator = generator
        self.local_model = local_model
        self.fixed_noise = generator.generate_noise_vector(16, device='cpu')
        self.resize = tfs.Resize(32)

        self.mean = torch.Tensor([0.5])
        self.std = torch.Tensor([0.5])

        self.transforms = torch.nn.Sequential(
            tfs.Resize(32),
            tfs.Normalize(mean=self.mean, std=self.std),
        )

    def denorm(self, x, channels=None, w=None, h=None, resize=False, device='cpu'):
        unnormalize = tfs.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist()).to(device)
        x = unnormalize(x)
        if resize:
            if channels is None or w is None or h is None:
                print('Number of channels, width and height must be provided for resize.')
            x = x.view(x.size(0), channels, w, h)
        return x

    def log_gan_images(self, caption, client_id, round_idx):
        images = make_grid(self.denorm(self.generator(self.fixed_noise)), nrow=8, padding=2, normalize=False,
                           range=None,
                           scale_each=False, pad_value=0)
        images = wandb.Image(images, caption=caption)
        wandb.log({f"Generator Outputs {client_id}": images, 'round': round_idx})

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
        generator, local_model = self.generator.to(device), self.local_model.to(device)

        if args.client_optimizer == "sgd":
            optimiser_G = torch.optim.SGD(self.generator.parameters(), lr=args.lr)
            optimiser_D = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)

        else:
            beta1, beta2 = 0.5, 0.999
            optimiser_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()),
                                           lr=args.lr,
                                           weight_decay=args.wd,
                                           amsgrad=True,
                                           betas=(beta1, beta2)
                                           )
            optimiser_D = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_model.parameters()),
                                           lr=args.lr,
                                           weight_decay=args.wd,
                                           amsgrad=True,
                                           betas=(beta1, beta2)
                                           )

        labelled_data, unlabelled_data = train_data

        self._gan_training(generator, local_model, labelled_data, unlabelled_data, args.epochs, optimiser_G,
                           optimiser_D, device)
        # self._train_loop(generator, train_data, None, args.epochs, optimiser_D, device)

    def _discriminator_output(self, logits):
        Z_x = torch.logsumexp(logits, dim=-1)
        return torch.sigmoid(Z_x / (Z_x + 1))

    def _gan_training(self, generator, discriminator, labelled_data, unlabelled_data, epochs, optimizer_G, optimizer_D,
                      device):
        generator.train()
        discriminator.train()

        real_label, fake_label = 1, 0  # Soft labels

        # train_data = labelled_data if unlabelled_data is None else zip(labelled_data, unlabelled_data)

        # Initialize BCELoss function
        unsupervised_loss = nn.BCELoss().to(device)
        supervised_loss = nn.CrossEntropyLoss().to(device)
        torch.autograd.set_detect_anomaly(True)
        transforms = self.transforms.to(device)

        epoch_loss_D = []
        epoch_loss_G = []
        for epoch in range(epochs):
            batch_loss_D, batch_loss_G = [], []
            # train_data = labelled_data if unlabelled_data is None else zip(labelled_data, unlabelled_data)
            for batch_idx, data in enumerate(
                    labelled_data if unlabelled_data is None else zip(labelled_data, cycle(unlabelled_data))):
                if unlabelled_data is not None:
                    (real, labels), (synth, synth_labels) = data
                    # ulreal = ulreal[0]  # zip packs iterables of varying length in to tuples so ulreal becomes [ulreal]
                    real, labels, synth, synth_labels = real.to(device), labels.to(device), synth.to(device), synth_labels.to(device)
                    real = torch.unsqueeze(real, 1) if len(real.shape) < 4 else real  # 1 channel datasets miss second dim
                    real = transforms(real)
                    with torch.no_grad():
                        real, labels = torch.cat((real, synth), dim=0), torch.cat((labels, synth_labels), dim=0)
                else:
                    real, labels = data
                    real, labels = real.to(device), labels.to(device)
                    real = transforms(real)
                    # unlabelled_real = real

                # unlabelled_real = unlabelled_real.to(device)
                b_size = real.size(0)

                generator.zero_grad()
                discriminator.zero_grad()
                """ Update Discriminator """

                # TRAIN THE DISCRIMINATOR (THE CLASSIFIER)
                optimizer_D.zero_grad()

                # 1. on Unlabelled data
                outputs = discriminator(real)
                logz_unlabel = torch.logsumexp(outputs, dim=-1)
                lossUL = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel)))

                # 2. on the generated data
                fake = generator.generate(b_size, device)
                outputs = discriminator(fake.detach())  # detach() because we are not training G here
                logz_fake = torch.logsumexp(outputs, dim=-1)
                lossD = 0.5 * torch.mean(F.softplus(logz_fake))

                # 3. on labeled data
                output = discriminator(real)
                logz_label = torch.logsumexp(output, dim=-1)
                prob_label = torch.gather(output, 1, labels.unsqueeze(1))
                labeled_loss = -torch.mean(prob_label) + torch.mean(logz_label)
                D_loss = labeled_loss + lossD + lossUL
                D_loss.backward()

                optimizer_D.step()

                # TRAIN THE DISCRIMINATOR (THE CLASSIFIER)
                optimizer_G.zero_grad()

                outputs = discriminator(fake)
                logz_unlabel = torch.logsumexp(outputs, dim=-1)
                G_loss = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel)))
                G_loss.backward()
                optimizer_G.step()

                # Save Losses for plotting later
                batch_loss_G.append(D_loss.item())
                batch_loss_D.append(G_loss.item())

                #######################################################################
                #                       ** END OF YOUR CODE **
                #######################################################################
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
        model = self.local_model.to(device)
        model.eval()

        transforms = self.transforms.to(device)

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
                x = transforms(x)
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
        generated_images = generator.generate(target_size, device=device)

        # Filter by realness score to select best generated images
        label_probs, labels = self._get_pseudo_labels_with_probability(discriminator(generated_images))
        mask = label_probs >= real_score_threshold
        good_generated_images, labels = generated_images[mask], labels[mask]

        # If generator is not good enough do not create synthetic dataset
        size = good_generated_images.size(0)
        if size == 0:
            return None, 0

        dataset = TensorDataset(good_generated_images, labels)
        # data_loader = DataLoader(dataset, batch_size=batch_size)
        return dataset, size
