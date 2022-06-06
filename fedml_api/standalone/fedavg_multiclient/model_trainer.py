import logging

import torch
from torch import nn

from knowledge_distillation.logits import Logits
from knowledge_distillation.soft_target import SoftTarget

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedAvgMultiClientModelTrainer(ModelTrainer):
    def __init__(self, model):
        """
        Args:
            adapter_model: Homogeneous model between clients that acts as knowledge transfer vehicle.
            local_model: Heterogeneous model that is chosen by clients that can better utilise client resources
        """
        super().__init__(model)
        self.model = model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args=None):
        """

        Args:
            train_data: Tuple of (train_data, kd_transfer_data).
            device: Device to perform training on
            args: Other args
        Returns:

        """
        model = self.model.to(device)

        optimizer = self.get_client_optimiser(model, args.client_optimizer, args.lr)
        # train and update assuming classification task
        cls_criterion = nn.CrossEntropyLoss().to(device)

        self._train_loop(model, train_data, cls_criterion, args.epochs, optimizer, device)

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

        epoch_loss = []
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

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
        return self.test_model(self.model, test_data, device, args)

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
        model = self.model.to(device)

        optimizer = self.get_client_optimiser(model, args.client_optimizer, args.lr)

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)

        # Transfer learning to private dataset
        self._train_loop(model, train_data=private_data, criterion=criterion, epochs=args.pretrain_epochs_private,
                         optimizer=optimizer, device=device)
