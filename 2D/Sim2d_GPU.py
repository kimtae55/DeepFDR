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

    def run(self, Lx, Ly, Lz, burn_in, n_iter, label):
        # Store P(theta=1) over time
        probOverTime = torch.zeros(burn_in)

        # initialize labels
        for i in range(Lx):
            for j in range(Ly):
                label[i][j] = torch.bernoulli(torch.tensor(0.5))  # p = 0.5 to choose 1

        print("c2")

        torch.save(label, '../testin.pt')

        iteration = 0
        iter_burn = 0
        while iteration < n_iter:
            # do stuff
            for i in range(Lx):
                for j in range(Ly):
                    conditional_distribution = self.model1_eq1(i, j, label, Lx)
                    label[i][j] = torch.bernoulli(conditional_distribution)

            if iter_burn < burn_in:
                probOverTime[iter_burn] = torch.count_nonzero(label) / (Lx * Ly)
                print("burn_in: ", iter_burn)
                iter_burn = iter_burn + 1
            else:
                iteration += 1

            torch.cuda.empty_cache()

        torch.save(label, '../testout.pt')

        print(probOverTime.shape)
        timex = torch.arange(0, burn_in, 1)
        plt.plot(timex[0::5], probOverTime[0::5], 'o')
        plt.savefig("test.png")

        return

    @staticmethod
    def in_range(i, j, size):
        return (0 <= i < size) and (0 <= j < size)

    def model1_eq1(self, i, j, label, size):
        beta = 0.8
        h = -2.5

        sum = 0
        if self.in_range(i, j + 1, size):
            sum += label[i][j + 1]
        if self.in_range(i, j - 1, size):
            sum += label[i][j - 1]
        if self.in_range(i + 1, j, size):
            sum += label[i + 1][j]
        if self.in_range(i - 1, j, size):
            sum += label[i - 1][j]
        num = torch.exp((beta * sum + h))
        denom = 1 + torch.exp(beta * sum + h)
        return num / denom

if __name__ == "__main__":
    np.random.seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    Lx = 15
    Ly = 15
    Lz = 15
    burn_in = 1000
    n_iter = 1

    print("GPU: ", torch.cuda.get_device_name(0))
    print("NUM ITERATIONS: ", burn_in)
    print("c0")

    sampler = GibbsSampler()
    label = torch.randn(Lx, Ly)
    print(label)
    label = label.cuda()
    print("c1")
    start = time.time()

    sampler.run(Lx, Ly, Lz, burn_in, n_iter, label)

    end = time.time()
    print("Time Elapsed:", end - start)