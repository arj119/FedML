import logging
import sys
import os

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_experiments.standalone.utils.config import parse_config
from fedml_experiments.standalone.utils.model import create_model
from fedml_experiments.standalone.utils.setup import setup

from fedml_api.standalone.federated_arjun.fedarjun_api import FedArjunAPI


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Fed Arjun arguments
    parser.add_argument('--kd_epochs', type=int, default=5,
                        help='Number of knowledge distillation epochs to be carried out in local training')

    parser.add_argument('--kd_lambda', type=float, default=1,
                        help='Weighting of knowledge distillation regularisation term')

    parser.add_argument('--transfer_set_percentage', type=float, default=0,
                        help='Percentage of local training data to be used for knowledge distillation.')

    parser.add_argument('--pretrain_epochs_private', type=int, default=10,
                        help='Number of pre-training epochs to be done by each client on the private dataset during'
                             ' transfer learning')

    parser.add_argument('--client_config_file', type=str,
                        default='./config/config.json',
                        help='Path to client model configuration. Should be a json file with list of '
                             '[(client_model, freq)]')

    return parser


if __name__ == "__main__":
    args, device, dataset = setup(algorithm_name='FedArjun', add_custom_args=add_args)
    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)

    client_model_config = parse_config(args.client_config_file)

    adapter_model = create_model(args, model_name=client_model_config['adapter_model'], output_dim=dataset[7])
    client_models = []
    client_num = 0

    for entry in client_model_config['client_models']:
        model = create_model(args, model_name=entry['model'], output_dim=dataset[7])
        client_models.append((model, entry['freq']))
        client_num += entry['freq']

    assert args.client_num_in_total == client_num

    # model = create_model(args, model_name=args.model, output_dim=dataset[7])
    # model_trainer = custom_model_trainer(args, model)
    logging.info(client_models)

    fedarjunAPI = FedArjunAPI(dataset, device, args, adapter_model, client_models)
    fedarjunAPI.train()
