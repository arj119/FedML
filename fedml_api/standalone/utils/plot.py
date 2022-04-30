import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import wandb

def convert_to_list(data: dict):
    """
    Args:
        data: Takes in label distribution dict which is {client_id -> {label -> count}}

    Returns:
        out: Dict of labels to counts {label -> [count]} where index of count corresponds to client id
        clients: List of client ids used
    """
    clients = list(data.keys())
    labels = set()
    for label_count in data.values():
        labels = labels.union(set(label_count.keys()))
    out = defaultdict(list)
    for l in labels:
        for c in clients:
            out[l].append(data[c].get(l, 0))

    return out, clients


def plot_label_distributions(data, alpha=0.5):
    label_count, clients = convert_to_list(data)

    for l, counts in label_count.items():
        plt.scatter(clients, [l] * len(clients), s=counts, label=f'$label={l}$', lw=1, c=['blue'])

    labels = list(label_count.keys())
    plt.yticks(np.arange(min(labels), max(labels) + 1, 1.0))
    plt.xticks(np.arange(min(clients), max(clients) + 1, 1.0))
    plt.ylabel('Training Label')
    plt.xlabel('Client ID')
    plt.title(f'Training Label Distribution $\\alpha = {alpha}$')
    plt.savefig('label_distribution.png')
    logging.info('----------- Saved Client Training Data Distribution ------------')
    image = wandb.Image('label_distribution.png')
    wandb.log({'Training Label Distribution': image})