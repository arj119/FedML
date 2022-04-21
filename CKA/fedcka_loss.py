from CKA.cka import CudaCKA
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Implemented from https://arxiv.org/pdf/2112.00407.pdf
"""


class FedCKALoss(nn.Module):
    def __init__(self, device, M=2):
        """
        Args:
            device: Device to run computation on
            M: Number of conv layers to perform loss on, defaults to 2 as from the paper referenced it states that early
            conv layers will have the most similarity regardless of dataset.
        """
        super().__init__()
        self.device = device
        self.cka = CudaCKA(device)
        self.M = M

    def forward(self, a_lt, a_lt_minus_one, a_gt):
        """
        Args:
            a_lt: Activations for current local model at conv layers
            a_lt_minus_one: Activations for previous local model (at time t-1) at conv layers
            a_gt: Activations for global model

        Returns:
            CKA loss defined in FedCKA
        """
        sum_val = 0

        for n in range(self.M):
            numerator = math.exp(self.cka.linear_CKA(a_lt[n], a_gt[n]))
            denominator = numerator + math.exp(self.cka.linear_CKA(a_lt[n], a_lt_minus_one[n]))

            sum_val += - math.log(numerator / denominator)

        return 1 / self.M * sum_val
