import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.model import create_model
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

    def experiment_start(self, client_model_config, args, device, dataset):
        client_num = client_model_config['client_num']
        assert args.client_num_in_total == client_num
        model = create_model(args, model_name=client_model_config['model'], output_dim=dataset[7])

        # model = create_model(args, model_name=args.model, output_dim=dataset[7])
        # model_trainer = custom_model_trainer(args, model)
        logging.info(model)

        api = FedAvgMultiClientAPI(dataset, device, args, model, client_num)
        api.train()


if __name__ == "__main__":
    FedAvgMultiClientExperiment().start()
