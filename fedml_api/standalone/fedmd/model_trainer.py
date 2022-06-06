import logging

import torch
from torch import nn

from knowledge_distillation.logits import Logits

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedMLModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def pre_train(self, public_data, private_data, device, args):
        """
        Pre-training in FedMD algorithm to do transfer learning from public data set
        to private dataset

        Args:
            public_data: Public data shared by all clients to perform KD on
            private_data: Private data only known to the client
            device: Device to perform training on
            args: Other args

        Returns:

        """
        model = self.model
        model.to(device)

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = self.get_client_optimiser(model, args.client_optimizer, args.lr)

        # Train on public dataset
        self._train_loop(model, train_data=public_data, criterion=criterion, epochs=args.pretrain_epochs_public,
                         optimizer=optimizer, device=device)
        # Transfer learning to private dataset
        self._train_loop(model, train_data=private_data, criterion=criterion, epochs=args.pretrain_epochs_private,
                         optimizer=optimizer, device=device)

    def train(self, train_data, device, args, public_data, consensus_logits):
        """

        Args:
            train_data: Private data only known to the client
            device: Device to perform training on
            public_data: Public data shared by all clients to perform KD on
            args: Other args
            consensus_logits: Aggregated logits at central server to be used as teacher knowledge

        Returns:

        """
        model = self.model
        model.to(device)

        # train and update assuming classification task
        kd_criterion = Logits()
        cls_criterion = nn.CrossEntropyLoss().to(device)

        optimizer = self.get_client_optimiser(model, args.client_optimizer, args.lr)

        # Digest: Each party trains its model f_k to approach the consensus_logits on the public dataset
        self._train_loop(model, public_data, cls_criterion, args.digest_epochs, optimizer, device, consensus_logits,
                         kd_criterion, args.kd_lambda)

        # Revisit: Each party trains its model fk on its own private data for a few epochs.
        self._train_loop(model, train_data, cls_criterion, args.revisit_epochs, optimizer, device)

    def _train_loop(self, model, train_data, criterion, epochs, optimizer, device, consensus_logits=None,
                    kd_criterion=None, kd_lambda=None):
        """

        Args:
            model: Model to be trained
            train_data: Training data
            criterion: Loss for main task
            epochs: Epochs of training to be completed
            optimizer: Optimiser that should be used to update hyperparams
            device: Device in which training should occur
            consensus_logits: Global logits from the server to align to in Digest stage
            kd_criterion: Knowledge distillation criterion
            kd_lambda: Knowledge distillation regularisation weight

        Returns:

        """

        model.train()

        if consensus_logits is not None:
            consensus_logits = consensus_logits.to(device)

        epoch_loss = []
        epoch_kd_loss = []
        epoch_task_loss = []
        for epoch in range(epochs):
            batch_loss = []
            consensus_logits_idx = 0
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

                optimizer.zero_grad()
                output = model(x)  # in classification case will be logits
                loss = criterion(output, labels)
                epoch_task_loss.append(loss.item())

                # Alignment with consensus logits if provided
                if not (kd_criterion is None or consensus_logits is None or kd_lambda is None):
                    target_logits = consensus_logits[consensus_logits_idx:consensus_logits_idx + len(x)]
                    consensus_logits_idx += len(x)
                    # Knowledge distillation regulariser
                    kd_loss = kd_lambda * kd_criterion(output, target_logits)
                    epoch_kd_loss.append(kd_loss.item())
                    loss += kd_loss

                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_kd_loss_avg = 1_000 if len(epoch_kd_loss) == 0 else sum(epoch_kd_loss) / len(epoch_kd_loss)
            logging.info(
                f'tEpoch: {epoch}\tLoss: {sum(epoch_loss) / len(epoch_loss):.6f} \tkd_loss: {epoch_kd_loss_avg:.6f} \ttask_loss: {sum(epoch_task_loss) / len(epoch_task_loss):.6f}')
        return epoch_loss

    def get_logits(self, public_data, device):
        """
        Calculate logits on public dataset. Part of communication phase

        Args:
            model: Model to evaluate public dataset on
            public_data: Public dataset

        Returns:

        """
        model = self.model
        model = model.to(device)
        model.eval()

        logits = []
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(public_data):
                x = x.to(device)
                logits.append(model(x))
        return torch.cat(logits).detach().cpu()

    def test(self, test_data, device, args=None):
        return self.test_model(self.model, test_data, device, args)

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
