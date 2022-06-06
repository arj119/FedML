import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.model import create_local_models_from_config, create_model
from fedml_api.standalone.fd_faug.FD_FAug_api import FDFAugAPI
from fedml_experiments.standalone.utils.experiment import ExperimentBase


class FDFAugExperiment(ExperimentBase):
    algorithm_name = 'FD_FAug with GAN Augmentation'

    def add_custom_args(self, parser):
        """
        parser : argparse.ArgumentParser
        return a parser added with args required by fit
        """
        # FD FAug arguments
        parser.add_argument('--kd_gamma', type=float, default=0.8,
                            help='Weighting of knowledge distillation regularisation term')

        parser.add_argument('--client_config_file', type=str,
                            default='./config/config.json',
                            help='Path to client model configuration. Should be a json file with list of '
                                 '[(client_model, freq)]')

        parser.add_argument('--share_percentage', type=float, default=0.05,
                            help='Number of epochs to be carries out on private dataset in local training')

        parser.add_argument('--nz', type=int,
                            default=100,
                            help='Latent vector size')

        parser.add_argument('--ngf', type=int,
                            default=100,
                            help='Number of generator features')

        parser.add_argument('--gen_lr', type=float, default=0.001,
                            help='Learning rate of generator')

        parser.add_argument('--gen_optimizer', type=str, default='adam',
                            help='Optimiser of generator')

        parser.add_argument('--faug_epochs', type=int,
                            default=200,
                            help='Number of FAug training epochs')
        return parser

    def experiment_start(self, client_model_config, client_models, args, device, dataset):
        generator = create_model(args, model_name=client_model_config['generator'], output_dim=dataset[7])
        discriminator = create_model(args, model_name=client_model_config['discriminator'], output_dim=dataset[7])
        api = FDFAugAPI(dataset, device, args, generator, discriminator, client_models)
        api.train()


if __name__ == "__main__":
    FDFAugExperiment().start()
