import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch
import numba
from numba import cuda, float64, int32, guvectorize, vectorize, config
from test_data_cpu import Data
import sys
import math

class GibbsSampler:
    def __init__(self, rng_seed):
        # RUN INSIDE MODEL FOLDER
        self.SAVE_DIR = os.path.join(os.getcwd(), '../data/model2/' + str(rng_seed) + '/label')
        self.LABEL_TXT_PATH = os.path.join(self.SAVE_DIR, 'label.txt')
        self.DISTRIBUTION_PNG = os.path.join(self.SAVE_DIR, 'distribution.png')

        # np.array([B_0, B_1, B_1d, B_2d, B_3d, B_1r, B_2r, B_3r])
        self.params = np.array([6e-3, 4e-6, 5e-4, 7e-3, 6e-3]).astype('float64') # ---> 30x30x30, ~0.18

        # Precomputed distance
        self.t_ij = np.load(os.path.join(os.getcwd(), 't_ij.npy'))
        self.theta_sum = np.load(os.path.join(os.getcwd(), 'theta_sum.npy'))
        return

    def run(self, burn_in, num_samples):
        # Store P(theta=1) over time
        print(self.params)
        probOverTime = np.zeros(burn_in)

        # initialize labels
        init_prob = np.full((Data.SIZE, Data.SIZE, Data.SIZE), 0.5)
        label = np.random.binomial(1, init_prob).astype('float64')

        iteration = 0
        iter_burn = 0
        while iteration < num_samples:
            # do sequential sampling using numba jit
            label = precompute_sampler(Data.SIZE, self.params, label, self.t_ij, self.theta_sum)
            if iter_burn < burn_in:
                print(label[0][0])
                probOverTime[iter_burn] = np.count_nonzero(label) / (Data.SIZE * Data.SIZE * Data.SIZE)
                print(str(iter_burn) + "_ratio: " +  str(probOverTime[iter_burn]))
                iter_burn += 1
            else:
                iteration += 1

        # Save the groundtruth label and distribution image
        if not os.path.isdir(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        with open(self.LABEL_TXT_PATH, 'w') as outfile:
            outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(label)))
            outfile.write('# params: {0}\n'.format(self.params))

            for data_slice in label:
                np.savetxt(outfile, data_slice, fmt='%-8.4f')
                outfile.write('# New z slice\n')

        timex = torch.arange(0, burn_in, 1)
        plt.plot(timex[0::5], probOverTime[0::5], 'o')
        plt.savefig(self.DISTRIBUTION_PNG)
        return

    def precompute_distance(self):
        start = time.time()
        t_ij = np.zeros((27000,30,30,30)).astype('float64')
        theta_sum = np.zeros((30,30,30))
        for i in range(30):
            print(i)
            for j in range(30):
                for k in range(30):
                    dist = d_ij(self.mat[0], self.mat[1], self.mat[2], float64(i), float64(j), float64(k))
                    t_ij[30*30*i + 30*j + k] = theta_ij(dist, self.params[1], self.params[2], self.params[3], self.params[4])
                    theta_sum[i][j][k] = np.sum(t_ij[30*30*i + 30*j + k])
        end = time.time()
        print("time: ", end - start)

        np.save(os.path.join(os.getcwd(), 't_ij.npy'), t_ij)
        np.save(os.path.join(os.getcwd(), 'theta_sum.npy'), theta_sum)

#======================================================================================================================#
# 1000 burn_in 15x15x15: 120s (cpu, no-python, vectorize)
@vectorize([float64(float64, float64, float64, float64, float64, float64)], target = 'cpu')
def d_ij(x, y, z, i, j, k):
    # create following guvectorize functions:
    # 1. d_ij
    # 2. rho_ij
    # 3. theta_ij
    # 4. theta_sum
    # 5. hj_theta_sum
    # test on gpu, see if it has speed improvement versus prange vs cpu vs parallel vs cuda
    return (((z - k) ** 2 + (y - j) ** 2 + (x - i) ** 2)**float64(0.5))


@vectorize([float64(float64, float64, float64, float64, float64)], target ='cpu')
def theta_ij(dist, B_1, B_1d, B_2d, B_3d):
    #return (B_1 + B_3d * dist*dist*dist + B_3r * rho*rho*rho + B_2d * dist*dist + B_2r * rho*rho + B_1d * dist + B_1r * rho)
    return B_1 + B_3d / ((1.0+dist)*(1.0+dist)*(1.0+dist)) + B_2d / ((1.0+dist)*(1.0+dist)) + B_1d / (1.0+dist)

@vectorize([float64(float64, float64)], target ='cpu')
def hj_theta(t_ij, result):
    return t_ij * result

@numba.njit(cache = True, parallel=True)
def precompute_sampler(voxel_size, params, label, t_ij, theta_sum):
    result = label
    for i in numba.prange(voxel_size):
        for j in numba.prange(voxel_size):
            for k in numba.prange(voxel_size):
                hj_t = hj_theta(t_ij[30*30*i + 30*j + k], result)
                hj_theta_sum = np.sum(hj_t)
                numerator = np.exp((-theta_sum[i][j][k] - params[0]) + hj_theta_sum)

                p = numerator / (1 + numerator)
                if np.isnan(p): #because lim inf/1+inf ~ 1
                    result[i][j][k] = np.random.binomial(1, 1.0)
                else:
                    result[i][j][k] = np.random.binomial(1, p)
    return result
 
if __name__ == "__main__":
    rng_seed = sys.argv[1]
    print("RAND: ", rng_seed)
    np.random.seed(int(rng_seed))
    numba.set_num_threads(int(numba.config.NUMBA_NUM_THREADS/2))
    burn_in = 1000
    num_samples = 1

    print("NUM ITERATIONS: ", burn_in)
    print("SIZE: ", Data.SIZE, Data.SIZE, Data.SIZE)

    start = time.time()
    sampler = GibbsSampler(rng_seed)
    sampler.run(burn_in, num_samples)
    end = time.time()
    print("time: ", end - start)
