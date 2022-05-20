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


def log_label_distribution_table(data: dict, class_num, dataset='Train'):
    """
       Args:
           data: Takes in label distribution dict which is {client_id -> {label -> count}}

   """
    all_labels = [str(l) for l in range(class_num)]
    columns = ['client_idx']
    columns.extend(all_labels)

    rows = []

    for client_idx, label_count in data.items():
        list_label_count = [label_count.get(l, 0) for l in range(class_num)]
        row = [client_idx]
        row.extend(list_label_count)
        rows.append(row)

    table = wandb.Table(columns=columns, data=rows)
    wandb.log({f'Client Dataset Label Distribution {dataset}': table})


def plot_label_distributions(data, class_num, alpha=None, dataset='Train'):
    plt.clf()
    label_count, clients = convert_to_list(data)
    log_label_distribution_table(data, class_num, dataset)
    for l, counts in label_count.items():
        plt.scatter(clients, [l] * len(clients), s=counts, label=f'$label={l}$', lw=1, c=['blue'])

    labels = list(label_count.keys())
    plt.yticks(np.arange(min(labels), max(labels) + 1, 1.0))
    plt.xticks(np.arange(min(clients), max(clients) + 1, 1.0))
    plt.ylabel('Label')
    plt.xlabel('Client ID')
    alpha_section = f"$\\alpha = {alpha}$"
    plt.title(f'Label Distribution ({dataset}) {alpha_section if alpha is not None else ""}')
    plt.savefig(f'{dataset}_label_distribution.png')
    logging.info('----------- Saved Client Training Data Distribution ------------')
    image = wandb.Image(f'{dataset}_label_distribution.png')
    wandb.log({f'Label Distribution {dataset}': image})
