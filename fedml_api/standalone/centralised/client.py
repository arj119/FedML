import logging

from fedml_api.standalone.utils.BaseClient import BaseClient


class CentralisedClient(BaseClient):

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                 device, model_trainer):
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, global_test_data, args,
                         device,
                         model_trainer)

        # self.local_training_data, sizes = self.prepare_local_training_data(local_training_data)
        self.local_training_data = local_training_data

        logging.info("self.local_sample_number = " + str(self.local_sample_number))
        logging.info(f"""
            ########### Client {client_idx}: ###############
            Local sample number: {self.local_sample_number}
            Training dataset size: {len(local_training_data)}
            Test dataset size: {len(local_test_data)}
            ################################################
        """)

    def train(self, w_global=None):
        logging.info(f'### Training Client {self.client_idx} ###')
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        logging.info(f'### Training Client {self.client_idx} (complete) ###')

    def pre_train(self):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(private_data=self.local_training_data, device=self.device, args=self.args)
