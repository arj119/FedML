import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.standalone.federated_arjun.fedarjun_api import FedArjunAPI
from fedml_experiments.standalone.utils.model import create_model
from fedml_experiments.standalone.utils.experiment import ExperimentBase


class FedrjunExperiment(ExperimentBase):
    algorithm_name = 'FedArjun'

    def add_custom_args(self, parser):
        """
        parser : argparse.ArgumentParser
        return a parser added with args required by fit
        """
        # Fed Arjun arguments
        parser.add_argument('--kd_epochs', type=int, default=5,
                            help='Number of knowledge distillation epochs to be carried out in local training')

        parser.add_argument('--kd_lambda', type=float, default=1,
                            help='Weighting of knowledge distillation regularisation term')

        parser.add_argument('--transfer_set_percentage', type=float, default=0,
                            help='Percentage of local training data to be used for knowledge distillation.')

        parser.add_argument('--pretrain_epochs_private', type=int, default=10,
                            help='Number of pre-training epochs to be done by each client on the private dataset during'
                                 ' transfer learning')

        parser.add_argument('--client_config_file', type=str,
                            default='./config/config.json',
                            help='Path to client model configuration. Should be a json file with list of '
                                 '[(client_model, freq)]')

        return parser

    def experiment_start(self, client_model_config, client_models, args, device, dataset):
        adapter_model = create_model(args, model_name=client_model_config['adapter_model'], output_dim=dataset[7])
        fedarjunAPI = FedArjunAPI(dataset, device, args, adapter_model, client_models)
        fedarjunAPI.train()


if __name__ == "__main__":
    FedrjunExperiment().start()
