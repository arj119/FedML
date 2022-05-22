import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.config import parse_config
from fedml_experiments.standalone.utils.model import create_model
from fedml_experiments.standalone.utils.setup import setup
from fedml_api.standalone.fedDTG.server import FedDTGAPI


def add_args(parser):
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

    parser.add_argument('--kd_alpha', type=float, default=0.6,
                        help='Weighting of knowledge distillation regularisation term')

    parser.add_argument('--kd_epochs', type=int, default=1,
                        help='Number of knowledge distillation epochs')

    return parser


if __name__ == "__main__":
    args, device, dataset = setup(algorithm_name='FedDTG', add_custom_args=add_args)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)

    client_model_config = parse_config(args.client_config_file)

    generator = create_model(args, model_name=client_model_config['generator'], output_dim=dataset[7])
    discriminator = create_model(args, model_name=client_model_config['discriminator'], output_dim=dataset[7])

    client_models = []
    client_num = 0

    for entry in client_model_config['client_models']:
        model = create_model(args, model_name=entry['model'], output_dim=dataset[7])
        client_models.append((model, entry['freq']))
        client_num += entry['freq']

    args.client_num_in_total = client_num

    # model = create_model(args, model_name=args.model, output_dim=dataset[7])
    # model_trainer = custom_model_trainer(args, model)
    logging.info(client_models)

    feddtgAPI = FedDTGAPI(dataset, device, args, generator, discriminator, client_models)
    feddtgAPI.train()
