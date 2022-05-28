import json


def create_argparser(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--dataset_r', type=float, default=0.25, metavar='R',
                        help='percentage of training dataset to use')

    parser.add_argument('--data_dir', type=str, default='./../../../data/mnist',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='PM',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.1, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=20, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=20, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=30,
                        help='how many round of communications we should use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--experiment_id', type=str, default='', help='Id to group experiments by')

    return parser


default_config = {
    'dataset': 'femnist',
    'data_dir': './../../../data/FederatedEMNIST',
    'partition_method': 'homo',  # hetero or homo
    'partition_alpha': ''
}


def parse_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        return merge_config(config)


def merge_config(config: dict):
    new_config = default_config.copy()
    new_config.update(config)
    return new_config
