import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.standalone.baseline.server import BaselineAPI
from fedml_experiments.standalone.utils.experiment import ExperimentBase


class BaselineExperiment(ExperimentBase):
    algorithm_name = 'Baseline'

    def add_custom_args(self, parser):
        """
        parser : argparse.ArgumentParser
        return a parser added with args required by fit
        """
        parser.add_argument('--pretrain_epochs_private', type=int, default=15,
                            help='Number of pre-training epochs to be done by each client on the private dataset during'
                                 ' transfer learning')

        parser.add_argument('--client_config_file', type=str,
                            default='./config/config.json',
                            help='Path to client model configuration. Should be a json file with list of '
                                 '[(client_model, freq)]')

        parser.add_argument('--share_percentage', type=float, default=0.0,
                            help='Percentage of private dataset each client is willing to share')

        return parser

    def experiment_start(self, client_model_config, client_models, args, device, dataset):
        api = BaselineAPI(dataset, device, args, client_models)
        api.train()


if __name__ == "__main__":
    BaselineExperiment().start()
