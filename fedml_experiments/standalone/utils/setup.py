import argparse
import logging
import wandb
import random
import numpy as np
import torch

from fedml_experiments.standalone.utils.config import create_argparser
from fedml_experiments.standalone.utils.dataset import load_data


def setup(algorithm_name, add_custom_args):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = create_argparser(argparse.ArgumentParser(description=f'{algorithm_name}-standalone'))
    parser = add_custom_args(parser)
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="fedml",
        name=f"{algorithm_name}", #-r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )
    wandb.define_metric("*", step_metric="Round")
    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_data(args, args.dataset)

    return args, device, dataset
