import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm


def gaussian_kernel(radius: int = 3, sigma: float = 4, device='cpu'):
    x_2 = np.linspace(-radius, radius, 2*radius+1) ** 2
    dist = np.sqrt(x_2.reshape(-1, 1) + x_2.reshape(1, -1)) / sigma
    kernel = norm.pdf(dist) / norm.pdf(0)
    kernel = torch.from_numpy(kernel.astype(np.float32))
    kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1]))

    if device == 'cuda':
        kernel = kernel.cuda()

    return kernel


class GaussianBlur2D(nn.Module):

    def __init__(self, radius: int = 2, sigma: float = 1):
        super(GaussianBlur2D, self).__init__()
        self.radius = radius
        self.sigma = sigma

    def forward(self, x):
        batch, nchan, nrow, ncol = x.shape
        kernel = gaussian_kernel(radius=self.radius, sigma=self.sigma, device=x.device.type)

        for c in range(nchan):
            x[:, c:c+1] = F.conv2d(x[:, c:c+1], kernel, padding=self.radius)

        return x

