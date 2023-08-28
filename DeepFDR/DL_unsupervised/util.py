# import collections

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import plotly.graph_objs as go

class EarlyStop:
    def __init__(self, patience: int = 10, threshold: float = 1e-2) -> None:
        # self.queue = collections.deque([0] * patience, maxlen=patience)
        self.patience = patience
        self.threshold = threshold
        self.wait = 0
        self.best_loss = np.Inf

    def __call__(self, train_loss: float) -> bool:
        """
        @monitor: value to monitor for early stopping
                  (e.g. train_loss, test_loss, ...)
        @mode: specify whether you want to maximize or minimize
               relative to @monitor
        """
        if np.less(self.threshold, 0):
            return False
        if train_loss is None:
            return False
        # self.queue.append(train_loss)
        if np.less(train_loss - self.best_loss, -self.threshold):
            self.best_loss = train_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False
 
def gaussian_kernel(radius: int = 3, sigma: float = 4, device='cpu'):
    x_2 = np.linspace(-radius, radius, 2*radius+1) ** 2
    dist = np.sqrt(x_2.reshape(-1, 1) + x_2.reshape(1, -1)) / sigma
    kernel = norm.pdf(dist) / norm.pdf(0)
    kernel = torch.from_numpy(kernel.astype(np.float32))
    kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1]))

    if device == 'cuda':
        kernel = kernel.cuda()

    return kernel

def gaussian_kernel_3d(radius: int = 3, sigma: float = 4, device='cpu'):
    x = np.linspace(-radius, radius, 2*radius+1)
    y = np.linspace(-radius, radius, 2*radius+1)
    z = np.linspace(-radius, radius, 2*radius+1)

    x, y, z = np.meshgrid(x, y, z)
    dist = np.sqrt(x**2 + y**2 + z**2)
    
    kernel = np.exp(-dist**2 / (2 * sigma**2))

    kernel = torch.from_numpy(kernel.astype(np.float32))
    kernel = kernel.view((1, 1, kernel.shape[0], kernel.shape[1], kernel.shape[2]))

    if device == 'cuda':
        kernel = kernel.cuda()

    return kernel

class NCutLoss3D(nn.Module):
    r"""Implementation of the continuous N-Cut loss, as in:
    'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)"""

    def __init__(self, radius: int = 4, sigma_1: float = 5, sigma_2: float = 1):
        r"""
        :param radius: Radius of the spatial interaction term
        :param sigma_1: Standard deviation of the spatial Gaussian interaction
        :param sigma_2: Standard deviation of the pixel value Gaussian interaction
        """
        super(NCutLoss3D, self).__init__()
        self.radius = radius
        self.sigma_1 = sigma_1  # Spatial standard deviation
        self.sigma_2 = sigma_2  # Pixel value standard deviation

    def forward(self, labels: Tensor, inputs: Tensor) -> Tensor:
        r"""Computes the continuous N-Cut loss, given a set of class probabilities (labels) and raw images (inputs).
        Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights
        relative to the class-wide average, rather than for every individual pixel.

        :param labels: Predicted class probabilities
        :param inputs: Raw images
        :return: Continuous N-Cut loss
        """
        num_classes = 2 # binary using sigmoid only returns 1, so coded this manually for now
        
        kernel = gaussian_kernel_3d(radius=self.radius, sigma=self.sigma_1, device=labels.device.type)
        loss = 0

        for k in range(num_classes):
            if k == 0: class_probs = labels[:,0].unsqueeze(1)
            elif k == 1: class_probs = 1.0 - labels[:,0].unsqueeze(1)
      
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3, 4), keepdim=True) / \
                (torch.mean(class_probs, dim=(2, 3, 4), keepdim=True) + 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))

            # Compute N-cut loss in 3D using conv3d
            numerator = torch.sum(class_probs * F.conv3d(class_probs * weights, kernel, padding=self.radius))
            denominator = torch.sum(class_probs * F.conv3d(weights, kernel, padding=self.radius))
            loss += nn.L1Loss()(numerator / (denominator + 1e-6), torch.zeros_like(numerator))

        return num_classes - loss


def visualize_3d_mesh(data):
    # Create a meshgrid of coordinates
    x, y, z = np.meshgrid(np.arange(data.shape[0]),
                          np.arange(data.shape[1]),
                          np.arange(data.shape[2]))

    # Create a scatter3d trace for the data points where value is 1
    scatter = go.Scatter3d(x=x[data == 1],
                           y=y[data == 1],
                           z=z[data == 1],
                           mode='markers',
                           marker=dict(size=2, color='blue')
                           )

    # Set layout properties
    layout = go.Layout(scene=dict(aspectmode='cube'))

    # Create a figure and add the scatter trace
    fig = go.Figure(data=[scatter], layout=layout)

    # Show the interactive plot
    fig.show()



