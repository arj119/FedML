import torch
import torch.nn as nn

import functools
import operator


class CNNParameterised(torch.nn.Module):
    """
    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, in_channels, out_classes, layers_shape, dropout, input_dim):
        super(CNNParameterised, self).__init__()
        self.net = nn.Sequential()
        for i, l_size in enumerate(layers_shape):
            layer = self._block(in_channels, out_channels=l_size, kernel_size=3, stride=2, padding=1, dropout=dropout)
            in_channels = l_size
            self.net.add_module(name=f'layer_{i}', module=layer)

        num_features_before_fcnn = functools.reduce(operator.mul, list(self.net(torch.rand(1, *input_dim)).shape))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features_before_fcnn, 128),
            nn.Linear(128, out_classes),
            # nn.Softmax()
        )

        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features_before_fcnn, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # self.net.add_module(name='classifier', module=self.classifier)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.25):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            # nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout)
        )

    def forward(self, x, discriminator=False):
        if len(x.shape) < 4:
            x = torch.unsqueeze(x, 1)
        x = self.net(x)

        if discriminator:
            return self.classifier(x), self.discriminator(x)
        else:
            return self.classifier(x)


def CNNSmall(in_channels, output_dim, input_dim, layers=None):
    return CNNParameterised(in_channels=in_channels, out_classes=output_dim, layers_shape=[8, 8], dropout=0.5,
                            input_dim=input_dim)


def CNNMedium(in_channels, output_dim, input_dim, layers=None):
    return CNNParameterised(in_channels=in_channels, out_classes=output_dim, layers_shape=[8, 16, 16], dropout=0.5,
                            input_dim=input_dim)


def CNNLarge(in_channels, output_dim, input_dim, layers=None):
    return CNNParameterised(in_channels=in_channels, out_classes=output_dim, layers_shape=[32, 32, 32],
                            dropout=0.5, input_dim=input_dim)


def CNNCustomLayers(in_channels, output_dim, input_dim, layers):
    return CNNParameterised(in_channels=in_channels, out_classes=output_dim, layers_shape=layers,
                            dropout=0.5, input_dim=input_dim)
