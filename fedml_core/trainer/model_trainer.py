from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """

    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, device, args=None):
        pass

    @abstractmethod
    def test(self, test_data, device, args=None):
        return self.test_model(self.model, test_data, device, args)

    @abstractmethod
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        pass

    def get_client_optimiser(self, model, optimiser_name, lr):
        if optimiser_name == "sgd":
            optimiser = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            beta1, beta2 = 0.5, 0.999
            optimiser = torch.optim.Adam(model.parameters(),
                                         lr=lr,
                                         eps=1e-08,
                                         weight_decay=1e-2,
                                         amsgrad=True,
                                         betas=(beta1, beta2)
                                         )
        return optimiser

    def test_model(self, model, test_data, device, args):
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

        y_pred = []
        y_true = []

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

                output = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
                y_pred.extend(output)  # Save Prediction
                target = target.data.cpu().numpy()
                y_true.extend(target)  # Save Truth

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics['test_total'] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics['test_total'] += target.size(0) * target.size(1)
        return metrics, y_pred, y_true
