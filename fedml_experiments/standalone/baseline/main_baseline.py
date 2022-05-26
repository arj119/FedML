import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.model import create_model
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

        return parser

    def experiment_start(self, client_model_config, args, device, dataset):
        client_models = []
        client_num = 0
        for entry in client_model_config['client_models']:
            model = create_model(args, model_name=entry['model'], output_dim=dataset[7])
            client_models.append((model, entry['freq']))
            client_num += entry['freq']

        args.client_num_in_total = client_num

        logging.info(client_models)

        api = BaselineAPI(dataset, device, args, client_models)
        api.train()


if __name__ == "__main__":
    BaselineExperiment().start()
