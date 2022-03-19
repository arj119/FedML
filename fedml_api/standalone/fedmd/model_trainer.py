import logging

import torch
from torch import nn

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

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

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
        kd_criterion = nn.MSELoss().to(device)
        cls_criterion = nn.CrossEntropyLoss().to(device)

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

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
        for epoch in range(epochs):
            batch_loss = []
            consensus_logits_idx = 0
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

                model.zero_grad()
                output = model(x)  # in classification case will be logits
                loss = criterion(output, labels)

                # Alignment with consensus logits if provided
                if not(kd_criterion is None or consensus_logits is None or kd_lambda is None):
                    target_logits = consensus_logits[consensus_logits_idx:consensus_logits_idx + len(x)]
                    consensus_logits_idx += len(x)
                    # Knowledge distillation regulariser
                    kd_loss = kd_lambda * kd_criterion(output, target_logits)
                    loss += kd_loss

                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(f'tEpoch: {epoch}\tLoss: {sum(epoch_loss) / len(epoch_loss):.6f}')
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

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
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
