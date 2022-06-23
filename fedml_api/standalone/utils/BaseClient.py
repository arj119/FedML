import logging
from torch.utils.data import DataLoader
from collections import Counter
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb


class BaseClient:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                 device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data: DataLoader = local_training_data
        self.local_test_data: DataLoader = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        self.global_val_data = None
        self.global_test_data = global_test_data

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, global_val_data,
                             local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.global_val_data = global_val_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        logging.info(f'### Training Client {self.client_idx} ###')
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        logging.info(f'### Training Client {self.client_idx} (complete) ###')
        return weights

    def local_test(self, data='test', class_num=None, round_idx=None):
        if data == 'test':
            test_data = self.local_test_data
        elif data == 'val':
            test_data = self.global_val_data
        elif data == 'global_test':
            test_data = self.global_test_data
        else:
            test_data = self._get_training_data_from_tuple()
        metrics, y_pred, y_true = self.model_trainer.test(test_data, self.device, self.args)

        # Build confusion matrix
        logging.info(f'Creating confusion matrix {self.client_idx}: {class_num}')
        cf_matrix = confusion_matrix(y_true, y_pred, labels=range(class_num), normalize='true')
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=range(class_num), columns=range(class_num))
        plt.figure(figsize=(10, 10))
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})
        plt.title(f'Client {self.client_idx} Confusion Matrix ({data.title()}): Round {round_idx}', fontsize=15)
        plt.xlabel('Ground Truth Label')
        plt.ylabel('Predicted Label')
        file_name = f'conf_matrix_{self.client_idx}.png'
        plt.savefig(file_name)
        image = wandb.Image(file_name)
        wandb.log({f'Client {self.client_idx}/{data.title()}/Confusion Matrix': image, 'Round': round_idx})
        plt.close('all')
        return metrics

    def get_label_distribution(self, mode='train'):
        if mode == 'train':
            data = self._get_training_data_from_tuple()
        else:
            data = self.local_test_data
        train_classes = list(torch.concat([label for _, label in data], dim=0).cpu().numpy())
        return self.client_idx, dict(Counter(train_classes))

    def _get_training_data_from_tuple(self):
        return self.local_training_data[0] if isinstance(self.local_training_data,
                                                         tuple) else self.local_training_data

    def get_dataset_size(self, dataset='train'):
        if dataset == 'train':
            set = self._get_training_data_from_tuple()
        elif dataset == 'test':
            set = self.local_test_data
        else:
            raise NotImplementedError()

        if isinstance(set, torch.utils.data.DataLoader):
            size = len(set.dataset)
        else:
            size = len(set)

        return size
