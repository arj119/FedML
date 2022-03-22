import logging

from torch.utils.data import random_split, DataLoader

from fedml_api.standalone.federated_arjun.model_trainer import FedArjunModelTrainer


class FedArjunClient:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer: FedArjunModelTrainer):
        self.client_idx = client_idx
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

        self.local_training_data, sizes = self.prepare_local_training_data(local_training_data)

        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        logging.info(f"""
            ########### Client {client_idx}: ###############
            Local sample number: {self.local_sample_number}
            Training set size: {sizes[0]}
            Transfer set size: {sizes[1]}
            Training dataset size: {len(local_training_data.dataset)}
            Test dataset size: {len(local_test_data.dataset)}
            ################################################
        """)

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data, _ = self.prepare_local_training_data(local_training_data)
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def prepare_local_training_data(self, local_training_data):
        len_local_dataset = len(local_training_data.dataset)
        transfer_set_size = int(self.args.transfer_set_percentage * len_local_dataset)
        transfer, train = random_split(local_training_data.dataset,
                                       [transfer_set_size, len_local_dataset - transfer_set_size])
        return (DataLoader(train, batch_size=self.args.batch_size), DataLoader(transfer, batch_size=self.args.batch_size)), \
               (len_local_dataset - transfer_set_size, transfer_set_size)

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data[0]
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
