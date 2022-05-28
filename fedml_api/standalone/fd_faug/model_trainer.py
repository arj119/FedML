import logging

import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FDFAugModelTrainer(ModelTrainer):
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

    def train(self, train_data, global_average_label_logits: dict, device, args):
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
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = self.get_client_optimiser(model, args.client_optimizer, args.lr)

        model.train()

        if global_average_label_logits is not None:
            for l, logit in global_average_label_logits.items():
                global_average_label_logits[l] = logit.to(device)

        label_sum_logits = dict()
        label_counts = dict()
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            x: torch.Tensor
            labels: torch.Tensor
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

                model.zero_grad()
                output = model(x)  # in classification case will be logits
                loss = criterion(output, labels)

                # Global logits alignment using co-distillation if provided
                if global_average_label_logits is not None:
                    global_average_logits_per_label = torch.stack(
                        [global_average_label_logits[l.cpu().item()] for l in labels])
                    # Cross entropy loss with soft targets given by softmax function applied to global average logits per label
                    soft_targets = torch.softmax(global_average_logits_per_label, dim=1)
                    kd_loss = criterion(output, soft_targets)
                    loss += args.kd_gamma * kd_loss

                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())

                with torch.no_grad():
                    # Update label average logits
                    unique_labels, counts = labels.unique(return_counts=True)
                    for l, c in zip(unique_labels.cpu().numpy(), counts.cpu().numpy()):
                        indices = (labels == l).nonzero(as_tuple=True)[0].cpu().numpy()
                        sum_logits = output[indices]
                        sum_logits = sum_logits.sum(dim=0).cpu()
                        label_sum_logits[l] = label_sum_logits.get(l, torch.zeros_like(sum_logits)) + sum_logits
                        label_counts[l] = label_counts.get(l, 0) + c

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(f'tEpoch: {epoch}\tLoss: {sum(epoch_loss) / len(epoch_loss):.6f}')

        # Average label counts and send average label logits to server
        for l, c in label_counts.items():
            label_sum_logits[l] /= c

        return label_sum_logits

    def test(self, test_data, device, args=None):
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
