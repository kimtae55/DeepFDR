import numpy as np
from scipy.stats import bernoulli
import math
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import time
import torch

class GibbsSampler(object):
    def __init__(self):
        return

    def run(self, Lx, Ly, Lz, burn_in, n_iter):
        # Store P(theta=1) over time
        probOverTime = torch.tensor(np.zeros(burn_in))

        # initialize labels
        label = torch.tensor(np.zeros((Lx, Ly, Lz)))
        for i in range(Lx):
            for j in range(Ly):
                for k in range(Lz):
                    label[i][j][k] = torch.bernoulli(torch.tensor(0.5))  # p = 0.5 to choose 1
        torch.save(label, 'testin.pt')

        iteration = 0
        iter_burn = 0
        while iteration < n_iter:
            # do stuff
            for i in range(Lx):
                for j in range(Ly):
                    for k in range(Lz):
                        conditional_distribution = self.model1_eq1(i, j, k, label, Lx)
                        label[i][j] = torch.bernoulli(conditional_distribution)

            if iter_burn < burn_in:
                probOverTime[iter_burn] = torch.count_nonzero(label) / (Lx * Ly * Lz)
                iter_burn = iter_burn + 1
                print("burn in: ", iter_burn)
            else:
                iteration += 1

        torch.save(label, 'testout.pt')

        print(probOverTime.shape)
        timex = torch.arange(0, burn_in, 1)
        plt.plot(timex[0::5], probOverTime[0::5], 'o')
        plt.savefig("test.png")

        return

    @staticmethod
    def in_range(i, j, k, size):
        return (0 <= i < size) and (0 <= j < size) and (0 <= k < size)

    def model1_eq1(self, i, j, k, label, size):
        beta = 0.8
        h = -2.5

        sum = 0
        if self.in_range(i, j, k + 1, size):
            sum += label[i][j][k + 1]
        if self.in_range(i, j, k - 1, size):
            sum += label[i][j][k - 1]
        if self.in_range(i, j + 1, k, size):
            sum += label[i][j + 1][k]
        if self.in_range(i, j - 1, k, size):
            sum += label[i][j - 1][k]
        if self.in_range(i + 1, j, k, size):
            sum += label[i + 1][j][k]
        if self.in_range(i - 1, j, k, size):
            sum += label[i - 1][j][k]
        num = torch.exp((beta * sum + h))  # probability of theta = 1
        denom = 1 + torch.exp(beta * sum + h)
        return num / denom

if __name__ == "__main__":
    np.random.seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Lx = 30
    Ly = 30
    Lz = 30
    burn_in = 2000
    n_iter = 1

    print("GPU: ", torch.cuda.get_device_name(0))
    print("NUM ITERATIONS: ", burn_in)
    print("VOXEL_SIZE: :", Lx, Ly, Lz)

    sampler = GibbsSampler()

    start = time.time()

    sampler.run(Lx, Ly, Lz, burn_in, n_iter)

    end = time.time()
    print("Time Elapsed:", end - start)