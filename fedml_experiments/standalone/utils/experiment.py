import argparse
import logging
from abc import ABC, abstractmethod
import random

import torch
import wandb
from wandb.util import generate_id
import numpy as np

from fedml_experiments.standalone.utils.config import parse_config, create_argparser
from fedml_experiments.standalone.utils.dataset import load_data


class ExperimentBase(ABC):
    def __init__(self):
        logging.basicConfig()

    num_repetitions = 5

    @property
    @abstractmethod
    def algorithm_name(self):
        pass

    def start(self):
        args = self._load_args()
        group_id = f'{args.dataset}_alpha={args.partition_alpha}_r={args.dataset_r}_{args.experiment_id}'
        for i in range(args.experiment_repetitions):
            if args.experiment_repetitions == 1:
                dataset = self._load_dataset(args, args.partition_seed)
                args, device = self._setup(args, seed=args.partition_seed, group_id=group_id, load_dataset=False)
            else:
                args, device, dataset = self._setup(args, seed=i, group_id=group_id)
            client_model_config = parse_config(args.client_config_file)
            self.experiment_start(client_model_config, args, device, dataset)
            wandb.finish()

    @abstractmethod
    def add_custom_args(self, parser):
        pass

    @abstractmethod
    def experiment_start(self, client_model_config, args, device, dataset):
        pass

    def _load_args(self):
        parser = create_argparser(argparse.ArgumentParser(description=f'{self.algorithm_name}-standalone'))
        parser = self.add_custom_args(parser)
        args = parser.parse_args()
        logging.info(args)

        return args

    def _setup(self, args, seed, group_id=None, load_dataset=True):
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        logging.info(device)

        wandb.init(
            project="fedml",
            name=f"{self.algorithm_name}",
            config=args,
            group=group_id
        )

        wandb.define_metric("*", step_metric="Round")
        # Set the random seed. The np.random seed determines the dataset partition.
        # The torch_manual_seed determines the initial weight.
        # We fix these two, so that we can reproduce the result.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        # load data
        if load_dataset:
            dataset = load_data(args, args.dataset)
            return args, device, dataset
        else:
            return args, device

    def _load_dataset(self, args, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        return load_data(args, args.dataset)
