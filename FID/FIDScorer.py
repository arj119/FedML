from scipy import linalg

from FID.InceptionV3 import InceptionV3
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
import torch


class FIDScorer:
    def __init__(self):
        self.model = InceptionV3()

    def calculate_activation_statistics(self, images, device='cpu'):

        model = self.model.to(device)
        model.eval()

        activations = []
        for (batch, _) in images:
            if len(batch.shape) < 4:
                batch = torch.unsqueeze(batch, 1)

            if batch.size(1) == 1:
                # greyscale
                batch = batch.repeat(1, 3, 1, 1)

            batch = batch.to(device)
            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            act = pred.cpu().data.numpy().reshape(pred.size(0), -1)
            activations.append(act)

        act = np.concatenate(activations, axis=0)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def calculate_fid(self, images_real, images_fake, device):
        mu_1, std_1 = self.calculate_activation_statistics(images_real, device=device)
        mu_2, std_2 = self.calculate_activation_statistics(images_fake, device=device)

        """get fretched distance"""
        fid_value = self.calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
        return fid_value
