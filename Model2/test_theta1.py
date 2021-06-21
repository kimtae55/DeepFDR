import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch
import numba
from numba import cuda, float64, int32, guvectorize, vectorize, config
from test_data_cpu import Data
import sys
import h5py
import math

numba.config.THREADING_LAYER = 'threadsafe'

class GibbsSampler:
    def __init__(self, rng_seed):
        # RUN INSIDE MODEL FOLDER
        self.PRECOMPUTE_PATH = os.path.join(os.getcwd(), 'theta_ij.hdf5') # make the person input a tmp directory path?
        self.SAVE_DIR = os.path.join(os.getcwd(), '../data/model2/' + str(rng_seed) + '/label')
        self.LABEL_PATH = os.path.join(self.SAVE_DIR, 'label.pt')
        self.LABEL_TXT_PATH = os.path.join(self.SAVE_DIR, 'label.txt')
        self.DISTRIBUTION_PNG = os.path.join(self.SAVE_DIR, 'distribution.png')
        self.LOG_PATH = os.path.join(self.SAVE_DIR, 'log.txt')
        self.DIST_PATH = os.path.join(self.SAVE_DIR, 'dist.txt')
        # CONSTANTS
        self.low = -1.0
        self.high = 1.0
        # PARAMETERS
        # np.array([B_0, B_1, B_1d, B_2d, B_3d, B_1r, B_2r, B_3r])
        # need to make numerator = -1.7346
        self.params = np.array([8, 1e-3, -2e-1, 2.0, 6.0]).astype('float64') # ---> 30x30x30, ~0.2
        #self.params = np.array([1.2, -0.01, 0.001, 0.4e-5, 0.0, 0.0, 0.0, 0.0]).astype('float64') # ---> 15x15x15, ~0.2
        self.mat = np.zeros((3, Data.SIZE, Data.SIZE, Data.SIZE))
        for j_i in range(Data.SIZE):
            for j_j in range(Data.SIZE):
                for j_k in range(Data.SIZE):
                    self.mat[0, j_i, j_j, j_k] = j_i
                    self.mat[1, j_i, j_j, j_k] = j_j
                    self.mat[2, j_i, j_j, j_k] = j_k
        self.mat = self.mat.astype('float64')
        return

    def precompute(self):
        with h5py.File(self.PRECOMPUTE_PATH, 'w') as f:
            for i in range(Data.SIZE):
                print(i)
                for j in range(Data.SIZE):
                    for k in range(Data.SIZE):
                        # function for d_ij
                        # function for rho_ij
                        arr = np.zeros((Data.SIZE, Data.SIZE, Data.SIZE))
                        for j_i in range(Data.SIZE):
                            for j_j in range(Data.SIZE):
                                for j_k in range(Data.SIZE):
                                    d_ij = np.sqrt((j_k - k) ** 2 + (j_j - j) ** 2 + (j_i - i) ** 2)
                                    arr[j_i][j_j][j_k] = d_ij
                        # save to file
                        f.create_dataset(str(i) + "_" + str(j) + "_" + str(k), data=arr)
        return

    def find_param(self, burn_in, num_samples):
        for b0 in np.arange(-5,5,0.2):
            for b1 in np.arange(-5, 5, 0.2):
                for b1d in np.arange(-5, 5, 0.2):
                    for b2d in np.arange(-5, 5, 0.2):
                        for b3d in np.arange(-5, 5, 0.2):
                            probOverTime = np.zeros(burn_in)
                            # initialize labels
                            init_prob = np.full((Data.SIZE, Data.SIZE, Data.SIZE), 0.5)
                            label = bernoulli(init_prob).astype('float64')
                            self.params = np.array([b0, b1, b1d, b2d, b3d, 0.0, 0.0, 0.0]).astype('float64')
                            iter_burn = 0
                            while iter_burn < burn_in:
                                # do sequential sampling using numba jit
                                label = test(Data.SIZE, self.params, label, self.mat)
                                probOverTime[iter_burn] = np.count_nonzero(label) / (Data.SIZE * Data.SIZE * Data.SIZE)
                                print("iter: ", iter_burn)
                                iter_burn += 1
                            if probOverTime[iter_burn-1] > 0.1 and probOverTime[iter_burn] < 0.25:
                                print(self.params)


    def run(self, burn_in, num_samples):
        # Store P(theta=1) over time
        print(self.params)
        probOverTime = np.zeros(burn_in)

        # initialize labels
        init_prob = np.full((Data.SIZE, Data.SIZE, Data.SIZE), 0.5)
        label = bernoulli(init_prob).astype('float64')
        iter_burn = 0
        while iter_burn < burn_in:
            # do sequential sampling using numba jit
            label = gibbs_Sampler(Data.SIZE, self.params, label, self.mat)
            print(label[0][0])
            probOverTime[iter_burn] = np.count_nonzero(label) / (Data.SIZE * Data.SIZE * Data.SIZE)
            print(str(iter_burn) + "_ratio: " +  str(probOverTime[iter_burn]))
            iter_burn += 1


        # Now do parallel sampling
        #samples = parallelSampling(init_prob, num_samples)
        #label = samples[0]
        label = gibbs_Sampler(Data.SIZE, self.params, label, self.mat)

        # Save the groundtruth label and distribution image
        if not os.path.isdir(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        with open(self.LABEL_TXT_PATH, 'w') as outfile:
            outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(label)))
            for data_slice in label:
                np.savetxt(outfile, data_slice, fmt='%-8.4f')
                outfile.write('# New z slice\n')

        timex = torch.arange(0, burn_in, 1)
        plt.plot(timex[0::5], probOverTime[0::5], 'o')
        plt.savefig(self.DISTRIBUTION_PNG)
        return

@numba.jit(nopython=True)
def bernoulli(distribution):
    result = np.zeros(distribution.shape)
    for i in range(result.shape[0]):
        for j in range(result.shape[0]):
            for k in range(result.shape[0]):
                result[i][j][k] = np.random.binomial(1, distribution[i][j][k])
    return result

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

@vectorize([float64(float64)], target ='cpu')
def rho_ij(dist):
    return float64(0.5)**dist

@vectorize([float64(float64, float64, float64, float64, float64)], target ='cpu')
def theta_ij(dist, B_1, B_1d, B_2d, B_3d):
    #return (B_1 + B_3d * dist*dist*dist + B_3r * rho*rho*rho + B_2d * dist*dist + B_2r * rho*rho + B_1d * dist + B_1r * rho)
    return B_1 + B_3d / ((1.0+dist)*(1.0+dist)*(1.0+dist)) + B_2d / ((1.0+dist)*(1.0+dist)) + B_1d / (1.0+dist)

@vectorize([float64(float64, float64)], target ='cpu')
def hj_theta(t_ij, result):
    return t_ij * result

@numba.njit(cache = True, parallel=True)
def gibbs_Sampler(voxel_size, params, label, mat):
    result = label
    for i in numba.prange(voxel_size):
        for j in numba.prange(voxel_size):
            for k in numba.prange(voxel_size):

                dist = d_ij(mat[0], mat[1], mat[2], float64(i), float64(j), float64(k))
                #rho = rho_ij(dist)
                t_ij = theta_ij(dist, params[1], params[2], params[3], params[4])

                theta_sum = np.sum(t_ij)

                hj_t = hj_theta(t_ij, result)

                hj_theta_sum = np.sum(hj_t)
                #print((-theta_sum - params[0]) + hj_theta_sum)

                numerator = np.exp((-theta_sum - params[0]) + hj_theta_sum)

                p = numerator / (1 + numerator)
                if np.isnan(p): #because lim inf/1+inf ~ 1
                    result[i][j][k] = np.random.binomial(1, 1.0)
                else:
                    result[i][j][k] = np.random.binomial(1, p)
    return result
#======================================================================================================================#

@numba.njit(parallel=True)
def parallelSampling(sample_size, voxel_size, dist):
    samples = np.zeros((sample_size,voxel_size,voxel_size,voxel_size))
    for num in numba.prange(sample_size):
        label = np.zeros((voxel_size,voxel_size,voxel_size))
        for i in range(voxel_size):
            for j in range(voxel_size):
                for k in range(voxel_size):
                    theta_ij = dist[voxel_size*voxel_size*i + voxel_size*j + k] # np.sum is expensive...... 0.3 secdons --> 2 seconds
                    # so think about pre-computing using tensors, then using gpu for all parallel stuff i can pre-compute
                    # then use numba jit for only sequential binomial operations
                    theta_sum = 0.0
                    #theta_ij = dist[voxel_size*voxel_size*i + voxel_size*j + k]
                    #p = np.sum(theta_ij) / (1 + np.sum(theta_ij))
                    p = 5/(1+5)
                    label[i][j][k] = np.random.binomial(1, p)
        samples[num] = label
    return samples


if __name__ == "__main__":
    rng_seed = sys.argv[1]
    print("RAND: ", rng_seed)
    np.random.seed(int(rng_seed))
    numba.set_num_threads(int(numba.config.NUMBA_NUM_THREADS/2))
    #cuda.select_device(0)
    burn_in = 100
    num_samples = 1

    print("NUM ITERATIONS: ", burn_in)
    print("SIZE: ", Data.SIZE, Data.SIZE, Data.SIZE)

    start = time.time()
    sampler = GibbsSampler(rng_seed)
    sampler.run(burn_in, num_samples)
    end = time.time()
    print("time: ", end - start)


    '''
    start = time.time()
    data = np.empty((27000, 30,30,30))
    with h5py.File(sampler.PRECOMPUTE_PATH, 'r') as f:
        index = 0
        for i in range(30):
            for j in range(30):
                for k in range(30):
                    dist = f[str(i) + "_" + str(j) + "_" + str(k)]
                    data[index] = np.array(dist)
                    index = index + 1
    end = time.time()
    print("time: ", end - start)
    '''