import logging

import torch
from torch import nn

from knowledge_distillation.logits import Logits
from knowledge_distillation.soft_target import SoftTarget

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedArjunModelTrainer(ModelTrainer):
    def __init__(self, adapter_model, local_model, args, train_adapter_model_only=False):
        """
        Args:
            adapter_model: Homogeneous model between clients that acts as knowledge transfer vehicle.
            local_model: Heterogeneous model that is chosen by clients that can better utilise client resources
        """
        super().__init__(adapter_model)
        self.adapter_model = adapter_model
        self.local_model = local_model
        self.train_adapter_model_only = train_adapter_model_only

        if args.client_optimizer == "sgd":
            self.local_optimizer_kd = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)
            self.local_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)
            self.adapter_optimizer = torch.optim.SGD(self.adapter_model.parameters(), lr=args.lr)

        else:
            self.local_optimizer_kd = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_model.parameters()),
                                                       lr=args.lr,
                                                       weight_decay=args.wd, amsgrad=True)
            self.local_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_model.parameters()),
                                                    lr=args.lr,
                                                    weight_decay=args.wd, amsgrad=True)
            self.adapter_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.adapter_model.parameters()), lr=args.lr,
                weight_decay=args.wd, amsgrad=True)

    def get_model_params(self):
        return self.adapter_model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.adapter_model.load_state_dict(model_parameters)

    def train(self, train_data, device, args=None):
        """

        Args:
            train_data: Tuple of (train_data, kd_transfer_data).
            device: Device to perform training on
            args: Other args
        Returns:

        """
        adapter_model, local_model = self.adapter_model.to(device), self.local_model.to(device)

        # train_data, kd_transfer_data = train_data

        if args.client_optimizer == "sgd":
            local_optimizer_kd = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)
            local_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr)
            adapter_optimizer = torch.optim.SGD(self.adapter_model.parameters(), lr=args.lr)

        else:
            local_optimizer_kd = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_model.parameters()),
                                                  lr=args.lr,
                                                  weight_decay=args.wd, amsgrad=True)
            local_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.local_model.parameters()),
                                               lr=args.lr,
                                               weight_decay=args.wd, amsgrad=True)
            adapter_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.adapter_model.parameters()),
                                                 lr=args.lr,
                                                 weight_decay=args.wd, amsgrad=True)

        # train and update assuming classification task
        kd_criterion = SoftTarget(T=4).to(device)
        cls_criterion = nn.CrossEntropyLoss().to(device)

        if self.train_adapter_model_only:
            self._train_loop(adapter_model, train_data, cls_criterion, args.epochs, adapter_optimizer, device)
        else:
            logging.info(f'Transfer Knowledge from adapter model to local model')

            # 1. Transfer knowledge from adapter model to local model
            self._knowledge_distillation(adapter_model, local_model, train_data, args.kd_epochs, local_optimizer_kd,
                                         cls_criterion, kd_criterion, 0.5, device)

            logging.info(f'Train local model')

            # # 2. Train local model
            self._train_loop(local_model, train_data, cls_criterion, args.epochs, local_optimizer, device)

            logging.info(f'Transfer knowledge from local model to adapter model')
            # 3. Transfer knowledge from local model to adapter model
            self._knowledge_distillation(local_model, adapter_model, train_data, args.kd_epochs, adapter_optimizer,
                                         cls_criterion, kd_criterion, 0.5, device)

    def _knowledge_distillation(self, teacher_model, student_model, transfer_set, epochs, optimizer, criterion,
                                kd_criterion, kd_lambda, device):
        """
        Args:
            teacher_model: Teacher model in KD.
            student_model: Student model that will learn from the teacher.
            transfer_set: Dataset used to transfer knowledge from the teacher model to the student model.
            epochs: Number of epochs.
            optimizer: Optimiser used to update parameters
            device: Device to perform training on
            criterion: Task criterion
            kd_criterion: Knowledge distillation criterion
            kd_lambda: Knowledge distillation regularisation weight
        Returns:
        """
        teacher_model.eval()
        student_model.train()

        epoch_loss = []
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(transfer_set):
                x, labels = x.to(device), labels.to(device)

                optimizer.zero_grad()
                student_output = student_model(x)  # in classification case will be logits
                task_loss = criterion(student_output, labels)

                # Knowledge Distillation Loss
                teacher_output = teacher_model(x).detach()
                kd_loss = kd_criterion(teacher_output, student_output)

                loss = (1 - kd_lambda) * task_loss + kd_lambda * kd_loss

                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(f'tEpoch: {epoch}\tLoss: {sum(epoch_loss) / len(epoch_loss):.6f}')
        return epoch_loss

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
        model = self.adapter_model.to(device) if self.train_adapter_model_only else self.local_model.to(device)
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

        # Transfer learning to private dataset
        self._train_loop(model, train_data=private_data, criterion=criterion, epochs=args.pretrain_epochs_private,
                         optimizer=self.local_optimizer, device=device)
