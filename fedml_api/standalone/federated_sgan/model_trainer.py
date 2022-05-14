import logging

import torch
from torch import nn
import torchvision.transforms as tfs
from torchvision.utils import make_grid
import wandb
from torch.utils.data import TensorDataset, DataLoader

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

    def log_gan_images(self, caption, client_id):
        images = make_grid(self.denorm(self.generator(self.fixed_noise)), nrow=8, padding=2, normalize=False,
                           range=None,
                           scale_each=False, pad_value=0)
        images = wandb.Image(images, caption=caption)
        wandb.log({f"Generator Outputs {client_id}": images})

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

        real_label, fake_label = 1, 0 # Soft labels

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
                    labelled_data if unlabelled_data is None else zip(labelled_data, unlabelled_data)):
                if unlabelled_data is not None:
                    (real, labels), ulreal = data
                    ulreal = ulreal[0]  # zip packs iterables of varying length in to tuples so ulreal becomes [ulreal]
                    real, labels, ulreal = real.to(device), labels.to(device), ulreal.to(device)
                    real = torch.unsqueeze(real, 1) if len(
                        real.shape) < 4 else real  # 1 channel datasets miss second dim
                    real = transforms(real)
                    with torch.no_grad():
                        unlabelled_real = torch.cat((real, ulreal), dim=0)
                else:
                    real, labels = data
                    real, labels = real.to(device), labels.to(device)
                    real = transforms(real)
                    unlabelled_real = real

                unlabelled_real = unlabelled_real.to(device)
                # real = transforms(real)
                b_size = unlabelled_real.size(0)

                generator.zero_grad()
                discriminator.zero_grad()
                """ Update Discriminator """

                # Update supervised discriminator
                class_logits = discriminator(real)
                D_sup_loss = supervised_loss(class_logits, labels)

                # Unsupervised discriminator
                label_real_adv = torch.full((b_size,), real_label, device=device, requires_grad=False,
                                            dtype=torch.float)
                label_fake_adv = torch.full((b_size,), fake_label, device=device, requires_grad=False,
                                            dtype=torch.float)

                # update unsupervised discriminator (adv)
                # Real examples
                gan_logits_real = self._discriminator_output(discriminator(unlabelled_real))
                errD_real = unsupervised_loss(gan_logits_real, label_real_adv)

                # Fake examples
                fake = generator.generate(b_size, device)
                gan_logits_fake = self._discriminator_output(discriminator(fake.detach()))
                errD_fake = unsupervised_loss(gan_logits_fake.view(-1), label_fake_adv)

                D_unsup_loss = 0.5 * (errD_real + errD_fake)
                D_loss = D_sup_loss + D_unsup_loss
                D_loss.backward()
                optimizer_D.step()

                """ Update Generator """
                class_logits = discriminator(fake)
                gan_logits = self._discriminator_output(class_logits)

                G_loss = unsupervised_loss(gan_logits.view(-1), label_real_adv)
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
                loss = criterion(output, labels)
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

        # train_data, kd_transfer_data = private_data

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)

        # # Transfer learning to private dataset
        # self._train_loop(model, train_data=private_data, criterion=criterion, epochs=args.pretrain_epochs_private,
        #                  optimizer=self.local_optimizer, device=device)

    def generate_synthetic_dataset(self, size, batch_size):
        self.generator.eval()
        generated_images = self.generator.generate(size, device='cpu')
        dataset = TensorDataset(generated_images)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
