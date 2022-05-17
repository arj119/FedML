import torch.nn as nn
from abc import ABC
import torch
import math


class Generator(ABC, nn.Module):
    def __init__(self, nz=100, img_size=32):
        super().__init__()
        self.nz = nz
        self.img_size = img_size

    def generate_noise_vector(self, b_size, device):
        return torch.randn(b_size, self.nz, 1, 1, device=device)

    def generate(self, b_size, device):
        return self.forward(self.generate_noise_vector(b_size, device))

    # custom weights initialization called on netG and netD
    def weights_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            self.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            self.weight.data.normal_(1.0, 0.02)
            self.bias.data.fill_(0)


class ImageGenerator(Generator):
    def __init__(self, nz=100, ngf=64, nc=3, img_size=32):
        """

        Args:
            nz: Latent vector size
            ngf: Number of generator features
            nc: Number of image channels to produce
            img_size: Size of the image to produce should be a power of 2 and > 4
        """
        super().__init__(nz, img_size)

        self.num_blocks = math.ceil(math.log(img_size // 8, 2))  # Account for first and last blocks

        self.main = nn.Sequential(
            # Input: N x latent_vector_size x 1 x 1
            self._block(nz, ngf * (2 ** self.num_blocks), 4, 1, 0),  # img: 4x4
        )

        for i in range(self.num_blocks):
            num_features = ngf * (2 ** (self.num_blocks - i))
            self.main.add_module(f'block {i}', self._block(num_features, num_features // 2, 4, 2, 1))

        self.main.add_module('end', nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(), ))

        self.weights_init()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, noise):
        return self.main(noise)


class ConditionalImageGenerator(Generator):
    def __init__(self, num_classes, nz=100, ngf=64, nc=3, img_size=32, init_size=4):
        """

        Args:
            nz: Latent vector size
            ngf: Number of generator features
            nc: Number of image channels to produce
            img_size: Size of the image to produce should be a power of 2 and > 4
        """
        super().__init__(nz, img_size)
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, nz)

        self.init_size = init_size  # Initial size before upsampling

        self.num_blocks = math.ceil(math.log(img_size // 8, 2))  # Account for first and last blocks

        self.first_filter_size = ngf * (2 ** self.num_blocks)
        self.l1 = nn.Sequential(nn.Linear(nz, self.first_filter_size * self.init_size ** 2))

        self.main = nn.Sequential(
            # Input: N x latent_vector_size x 1 x 1
            # self._block(nz, ngf * (2 ** self.num_blocks), 4, 1, 0),  # img: 4x4
        )

        for i in range(self.num_blocks):
            num_features = ngf * (2 ** (self.num_blocks - i))
            self.main.add_module(f'block {i}', self._block(num_features, num_features // 2, 4, 2, 1))

        self.main.add_module('end', nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(), ))

        self.weights_init()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], self.first_filter_size, self.init_size, self.init_size)
        img = self.main(out)
        return img

    def generate_noise_vector(self, b_size, device):
        return torch.randn(b_size, self.nz, device=device)

    def generate_random_labels(self, b_size, device):
        return torch.randint(0, self.num_classes, (b_size,), device=device)

    def generate_balanced_labels(self, b_size, device):
        number_of_each_class = b_size // self.num_classes
        leftover = b_size % self.num_classes

        labels = []
        for c in range(self.num_classes):
            num = number_of_each_class
            if leftover > 0:
                num += 1
                leftover -= 1
            labels.append(torch.full((num,), c, device=device))

        return torch.cat(labels, dim=0)

    def generate(self, b_size, device):
        return self.forward(self.generate_noise_vector(b_size, device), self.generate_random_labels(b_size, device))
