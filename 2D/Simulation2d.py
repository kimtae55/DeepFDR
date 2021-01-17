import numpy as np
from scipy.stats import bernoulli
import math
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import time

def in_range(i, j, size):
    return (0 <= i < size) and (0 <= j < size)


def model1_eq1(i, j, label, size):
    beta = 0.8
    h = -2.5

    sum = 0
    if in_range(i, j + 1, size):
        sum += label[i][j + 1]
    if in_range(i, j - 1, size):
        sum += label[i][j - 1]
    if in_range(i + 1, j, size):
        sum += label[i + 1][j]
    if in_range(i - 1, j, size):
        sum += label[i - 1][j]
    num = math.exp((beta * sum + h))
    denom = 1 + math.exp(beta * sum + h)
    return num / denom


def run_gibbs(Lx,Ly,Lz,burn_in,n_iter):
    np.random.seed(12345)
    # Store P(theta=1) over time
    probOverTime = np.zeros(burn_in)

    # initialize labels
    label = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            label[i][j] = bernoulli.rvs(0.2)  # p = 0.5 to choose 1

    with open('../test.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(label.shape))
        outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(label)))
        for data_slice in label:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
            outfile.write('# New z slice\n')

    iteration = 0
    iter_burn = 0
    while iteration < n_iter:
        # do stuff
        for i in range(Lx):
            for j in range(Ly):
                conditional_distribution = model1_eq1(i, j, label, Lx)
                label[i][j] = bernoulli.rvs(conditional_distribution)

        if iter_burn < burn_in:
            probOverTime[iter_burn] = np.count_nonzero(label) / (Lx * Ly)
            iter_burn = iter_burn+1
        else:
            iteration += 1

    with open('../test1.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(label.shape))
        outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(label)))
        for data_slice in label:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
            outfile.write('# New z slice\n')

    print(probOverTime.shape)
    time = np.arange(0,burn_in,1)
    plt.plot(time[0::5],probOverTime[0::5],'o')
    plt.savefig("test.png")

if __name__ == "__main__":
    Lx = 15
    Ly = 15
    Lz = 15

    burn_in = 1000
    n_iter = 1

    start = time.time()

    run_gibbs(Lx, Ly, Lz, burn_in, n_iter)

    end = time.time()
    # 11.2345 seconds

    print("time elapsed:", end - start)
'''
    initdata = np.loadtxt('test.txt').reshape((Lx, Ly))
    data = np.loadtxt('test1.txt').reshape((Lx, Ly))

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(initdata)
    plt.title('initlabel')
    plt.subplot(1, 2, 2)
    plt.imshow(data)
    plt.title('label')
    plt.savefig('2d_2')
    plt.show()
'''