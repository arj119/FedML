import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.model import create_model, create_local_models_from_config
from fedml_api.standalone.fedgdkd.server import FedGDKDAPI
from fedml_experiments.standalone.utils.experiment import ExperimentBase


class FedGDKDExperiment(ExperimentBase):
    algorithm_name = 'FedGDKD'

    def add_custom_args(self, parser):
        """
        parser : argparse.ArgumentParser
        return a parser added with args required by fit
        """
        # Fed Arjun arguments
        parser.add_argument('--discriminator_epochs', type=int, default=1,
                            help='Number of epochs to train the generator with')

        parser.add_argument('--client_config_file', type=str,
                            default='./config/config.json',
                            help='Path to client model configuration. Should be a json file with list of '
                                 '[(client_model, freq)]')

        parser.add_argument('--nz', type=int,
                            default=100,
                            help='Latent vector size')

        parser.add_argument('--ngf', type=int,
                            default=100,
                            help='Number of generator features')

        parser.add_argument('--kd_alpha', type=float, default=0.8,
                            help='Weighting of knowledge distillation regularisation term')

        parser.add_argument('--gen_lr', type=float, default=0.001,
                            help='Learning rate of generator')

        parser.add_argument('--gen_optimizer', type=str, default='adam',
                            help='Optimiser of generator')

        parser.add_argument('--kd_epochs', type=int, default=5,
                            help='Number of knowledge distillation epochs')

        return parser

    def experiment_start(self, client_model_config, client_models, args, device, dataset):
        generator = create_model(args, model_name=client_model_config['generator'], output_dim=dataset[7])
        api = FedGDKDAPI(dataset, device, args, generator, client_models)
        api.train()


if __name__ == "__main__":
    FedGDKDExperiment().start()
