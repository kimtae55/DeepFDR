import numpy as np
from scipy.stats import bernoulli
import math
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import time

def in_range(i, j, k, size):
    return (0 <= i < size) and (0 <= j < size) and (0 <= k < size)


def model1_eq1(i, j, k, label, size):
    beta = 0.8
    h = -2.5

    sum = 0
    if in_range(i, j, k + 1, size):
        sum += label[i][j][k + 1]
    if in_range(i, j, k - 1, size):
        sum += label[i][j][k - 1]
    if in_range(i, j + 1, k, size):
        sum += label[i][j + 1][k]
    if in_range(i, j - 1, k, size):
        sum += label[i][j - 1][k]
    if in_range(i + 1, j, k, size):
        sum += label[i + 1][j][k]
    if in_range(i - 1, j, k, size):
        sum += label[i - 1][j][k]
    num = math.exp((beta * sum + h)) # probability of theta = 1
    denom = 1 + math.exp(beta * sum + h)
    return num / denom


def run_gibbs(Lx,Ly,Lz,burn_in,n_iter):
    np.random.seed(12345)

    # for plotting p(theta = 1) over time
    probOverTime = np.zeros(burn_in)

    # initialize labels
    label = np.zeros((Lx, Ly, Lz))
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                label[i][j][k] = bernoulli.rvs(0.5)  # p = 0.5 to choose 1

    with open('../data_15x15x15/init_15x15.txt', 'w') as outfile:
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
                for k in range(Lz):
                    conditional_distribution = model1_eq1(i, j, k, label, Lx)
                    label[i][j][k] = bernoulli.rvs(conditional_distribution)

        if iter_burn < burn_in:
            probOverTime[iter_burn] = np.count_nonzero(label) / (Lx * Ly * Lz)
            iter_burn += 1
            print("burn in: ", iter_burn)
        else:
            iteration += 1
            print("sampled: ", iteration)

    with open('../data_15x15x15/beta_02_iter_3000.txt', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(label.shape))
        outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(label)))
        for data_slice in label:
            np.savetxt(outfile, data_slice, fmt='%-7.2f')
            outfile.write('# New z slice\n')

    time = np.arange(0, burn_in, 1)
    plt.plot(time[0::5],probOverTime[0::5],'o')
    plt.savefig("../data_15x15x15/distribution_15x15x15.png")

if __name__ == "__main__":
    Lx = 15
    Ly = 15
    Lz = 15

    burn_in = 1000
    n_iter = 1

    start = time.time()

    run_gibbs(Lx, Ly, Lz, burn_in, n_iter)

    end = time.time()
    print("time elapsed:", end - start)


'''
    initdata = np.loadtxt('init30x30.txt').reshape((Lx, Ly, Lz))
    data = np.loadtxt('beta_02_iter_3000.txt').reshape((Lx, Ly, Lz))
    x, y, z = data.nonzero()
    df = pd.DataFrame(np.hstack((x[:, None], y[:, None], z[:, None])),columns=['z','x','y'])
    print(df)
    fig = px.scatter_3d(df,x = 'z',y = 'x',z='y', opacity=0.8)
    fig.show()

    for k in range(Lx):
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(initdata[k])
        plt.title('initlabel')
        plt.subplot(1, 2, 2)
        plt.imshow(data[k])
        plt.title('label')
        plt.show()
'''
