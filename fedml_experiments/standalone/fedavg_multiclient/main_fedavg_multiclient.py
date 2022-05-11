import logging
import sys
import os

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.config import parse_config
from fedml_experiments.standalone.utils.model import create_model
from fedml_experiments.standalone.utils.setup import setup
from fedml_api.standalone.fedavg_multiclient.fedavgmulticlient_api import FedAvgMultiClientAPI


def add_args(parser):
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


if __name__ == "__main__":
    args, device, dataset = setup(algorithm_name='FedAvg', add_custom_args=add_args)
    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)

    client_model_config = parse_config(args.client_config_file)

    model = create_model(args, model_name=client_model_config['model'], output_dim=dataset[7])
    client_num = client_model_config['client_num']
    args.client_num_in_total = client_num

    # model = create_model(args, model_name=args.model, output_dim=dataset[7])
    # model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    api = FedAvgMultiClientAPI(dataset, device, args, model, client_num)
    api.train()
