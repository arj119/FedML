import logging

import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FDFAugModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, global_average_label_logits: dict, device, args):
        """

        Args:
            train_data: Private data only known to the client
            device: Device to perform training on
            public_data: Public data shared by all clients to perform KD on
            args: Other args
            consensus_logits: Aggregated logits at central server to be used as teacher knowledge

        Returns:

        """
        model = self.model
        model.to(device)

        # train and update assuming classification task
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = self.get_client_optimiser(model, args.client_optimizer, args.lr)

        model.train()

        if global_average_label_logits is not None:
            for l, logit in global_average_label_logits.items():
                global_average_label_logits[l] = logit.to(device)

        label_sum_logits = dict()
        label_counts = dict()
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []

            x: torch.Tensor
            labels: torch.Tensor
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

                model.zero_grad()
                output = model(x)  # in classification case will be logits
                loss = criterion(output, labels)

                # Global logits alignment using co-distillation if provided
                if global_average_label_logits is not None:
                    global_average_logits_per_label = torch.stack(
                        [global_average_label_logits[l.cpu().item()] for l in labels])
                    # Cross entropy loss with soft targets given by softmax function applied to global average logits per label
                    soft_targets = torch.softmax(global_average_logits_per_label, dim=1)
                    kd_loss = criterion(output, soft_targets)
                    loss = (1 - args.kd_gamma) * loss + args.kd_gamma * kd_loss

                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())

                with torch.no_grad():
                    # Update label average logits
                    unique_labels, counts = labels.unique(return_counts=True)
                    for l, c in zip(unique_labels.cpu().numpy(), counts.cpu().numpy()):
                        indices = (labels == l).nonzero(as_tuple=True)[0].cpu().numpy()
                        sum_logits = output[indices]
                        sum_logits = sum_logits.sum(dim=0).cpu()
                        label_sum_logits[l] = label_sum_logits.get(l, torch.zeros_like(sum_logits)) + sum_logits
                        label_counts[l] = label_counts.get(l, 0) + c

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(f'tEpoch: {epoch}\tLoss: {sum(epoch_loss) / len(epoch_loss):.6f}')

        # Average label counts and send average label logits to server
        for l, c in label_counts.items():
            label_sum_logits[l] /= c

        return label_sum_logits

    def test(self, test_data, device, args=None):
        return self.test_model(self.model, test_data, device, args)

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
