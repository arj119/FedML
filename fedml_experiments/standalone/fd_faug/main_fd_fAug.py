import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.config import parse_config
from fedml_experiments.standalone.utils.model import create_model
from fedml_experiments.standalone.utils.setup import setup

from fedml_api.standalone.fd_faug.FD_FAug_api import FDFAugAPI


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # FD FAug arguments
    parser.add_argument('--kd_gamma', type=float, default=1.,
                        help='Weighting of knowledge distillation regularisation term')

    parser.add_argument('--client_config_file', type=str,
                        default='./config/config.json',
                        help='Path to client model configuration. Should be a json file with list of '
                             '[(client_model, freq)]')
    return parser


if __name__ == "__main__":
    args, device, dataset = setup(algorithm_name='FD_FAug', add_custom_args=add_args)
    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)

    client_model_config = parse_config(args.client_config_file)

    client_models = []
    client_num = 0
    for entry in client_model_config['client_models']:
        model = create_model(args, model_name=entry['model'], output_dim=dataset[7])
        client_models.append((model, entry['freq']))
        client_num += entry['freq']

    args.client_num_in_total = client_num

    logging.info(client_models)

    fd_faugAPI = FDFAugAPI(dataset, device, args, client_models)
    fd_faugAPI.train()
