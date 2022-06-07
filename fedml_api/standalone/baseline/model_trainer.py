import logging

import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class BaselineModelTrainer(ModelTrainer):
    def __init__(self, local_model, args):
        """
        Args:
            adapter_model: Homogeneous model between clients that acts as knowledge transfer vehicle.
            local_model: Heterogeneous model that is chosen by clients that can better utilise client resources
        """
        super().__init__(None)
        self.local_model = local_model

        if args.client_optimizer == "sgd":
            self.local_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)

        else:
            self.local_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_model.parameters()),
                                                    lr=args.lr,
                                                    weight_decay=args.wd, amsgrad=True)

    def get_model_params(self):
        return None

    def set_model_params(self, model_parameters):
        pass

    def train(self, train_data, device, args=None):
        """

        Args:
            train_data: Tuple of (train_data, kd_transfer_data).
            device: Device to perform training on
            args: Other args
        Returns:

        """
        local_model = self.local_model.to(device)

        if args.client_optimizer == "sgd":
            local_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)

        else:
            local_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_model.parameters()),
                                               lr=args.lr,
                                               weight_decay=args.wd, amsgrad=True)

        # train and update assuming classification task
        cls_criterion = nn.CrossEntropyLoss().to(device)

        self._train_loop(local_model, train_data, cls_criterion, args.epochs, local_optimizer, device)

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

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)

        # Transfer learning to private dataset
        self._train_loop(model, train_data=private_data, criterion=criterion, epochs=args.pretrain_epochs_private,
                         optimizer=self.local_optimizer, device=device)
