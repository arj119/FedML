import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.model import create_model, create_local_models_from_config
from fedml_api.standalone.fedmd.FedMD_api import FedMDAPI
from fedml_experiments.standalone.utils.experiment import ExperimentBase


class FedMDExperiment(ExperimentBase):
    algorithm_name = 'FedMD'

    def add_custom_args(self, parser):
        """
        parser : argparse.ArgumentParser
        return a parser added with args required by fit
        """
        # FedMD arguments
        parser.add_argument('--public_dataset_size', type=int, default=5000,
                            help='Size of the public dataset that should be created')

        parser.add_argument('--pretrain_epochs_public', type=int, default=20,
                            help='Number of pre-training epochs to be done by each client on the public dataset')

        parser.add_argument('--pretrain_epochs_private', type=int, default=5,
                            help='Number of pre-training epochs to be doen by each client on the private dataset during'
                                 ' transfer learning')

        parser.add_argument('--digest_epochs', type=int, default=5,
                            help='Number of knowledge distillation epochs to be carried out in local training')

        parser.add_argument('--kd_lambda', type=float, default=0.5,
                            help='Weighting of knowledge distillation regularisation term')

        parser.add_argument('--revisit_epochs', type=int, default=5,
                            help='Number of epochs to be carries out on private dataset in local training')

        parser.add_argument('--share_percentage', type=float, default=0.05,
                            help='Number of epochs to be carries out on private dataset in local training')

        parser.add_argument('--client_config_file', type=str,
                            default='./config/config.json',
                            help='Path to client model configuration. Should be a json file with list of '
                                 '[(client_model, freq)]')
        return parser

    def experiment_start(self, client_model_config, client_models, args, device, dataset):
        api = FedMDAPI(dataset, device, args, client_models)
        api.train()


if __name__ == "__main__":
    FedMDExperiment().start()
