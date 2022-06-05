import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.model import create_model, create_local_models_from_config
from fedml_api.standalone.fedavg_multiclient.fedavgmulticlient_api import FedAvgMultiClientAPI
from fedml_experiments.standalone.utils.experiment import ExperimentBase


class FedAvgMultiClientExperiment(ExperimentBase):
    algorithm_name = 'FedAvg'

    def add_custom_args(self, parser):
        """
        parser : argparse.ArgumentParser
        return a parser added with args required by fit
        """
        # Fed Arjun arguments
        parser.add_argument('--client_config_file', type=str,
                            default='./config/config.json',
                            help='Path to client model configuration. Should be a json file with list of '
                                 '[(client_model, freq)]')

        return parser

    def experiment_start(self, client_model_config, client_models, args, device, dataset):
        assert len(
            client_models) == 1, f'FedAvg can only work with a single shared model, you have provided {len(client_models)}'

        model, num_clients = client_models[0]
        api = FedAvgMultiClientAPI(dataset, device, args, model, num_clients)
        api.train()


if __name__ == "__main__":
    FedAvgMultiClientExperiment().start()
