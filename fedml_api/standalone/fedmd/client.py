from fedml_api.standalone.utils.BaseClient import BaseClient


class Client(BaseClient):
    def train(self, public_data, consensus_logits):
        self.model_trainer.train(self.local_training_data, self.device, self.args, public_data, consensus_logits)

    def pre_train(self, public_data):
        """

        Args:
            public_data: Public dataset used for transfer learning

        Returns:

        """
        self.model_trainer.pre_train(public_data=public_data, private_data=self.local_training_data, device=self.device,
                                     args=self.args)

    def get_logits(self, public_data):
        return self.model_trainer.get_logits(public_data, self.device)
