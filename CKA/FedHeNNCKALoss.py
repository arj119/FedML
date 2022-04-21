from CKA.cka import CudaCKA
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Implemented from https://arxiv.org/pdf/2202.07757.pdf
"""


class FedHeNNCKALoss(nn.Module):
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

    def forward(self, local_out, global_out):
        """

        Args:
            local_out: Local representation output (all but last layer)
            global_out: Global representation output (all but last layer)

        Returns:

        """
        return self.cka.linear_CKA(local_out, global_out)