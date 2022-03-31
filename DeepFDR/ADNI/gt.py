import numpy as np
import os
import matplotlib.pyplot as plt
import time
import numba
from numba import cuda, float64, int32, guvectorize, vectorize, config
import sys
import math
import argparse

class GibbsSampler:
    def __init__(self, rng_seed, datapath):
        # RUN INSIDE MODEL FOLDER
        self.SIZE = 15
        self.SAVE_DIR = os.path.join(datapath, str(rng_seed), 'label')
        self.LABEL_NPY_PATH = os.path.join(self.SAVE_DIR, 'label.npy')
        self.LABEL_TXT_PATH = os.path.join(self.SAVE_DIR, 'label.txt')
        self.DISTRIBUTION_PNG = os.path.join(self.SAVE_DIR, 'distribution.png')

        # np.array([B_0, B_1, B_1d, B_2d, B_3d, B_1r, B_2r, B_3r])
        self.params = np.array([6e-3, 4e-6, 5e-4, 7e-3, 6e-3]).astype('float64') # ---> 30x30x30, ~0.18

        # Precomputed distance
        self.dist = np.memmap('distance.npy', dtype='float32', mode='r', shape=(self.SIZE**3,self.SIZE**3)) 

        return

    def run(self, burn_in, num_samples):
        # Store P(theta=1) over time
        probOverTime = np.zeros(burn_in)

        # initialize labels
        init_prob = np.full((self.SIZE**3), 0.5)
        label = np.random.binomial(1, init_prob).astype('float64')

        iteration = 0
        iter_burn = 0
        while iteration < num_samples:
            # do sequential sampling using numba jit
            label = sample(self.SIZE**3, self.params, label, self.dist)
            if iter_burn < burn_in:
                print('burn_in: ', iter_burn)
                probOverTime[iter_burn] = np.count_nonzero(label) / (self.SIZE * self.SIZE * self.SIZE)
                iter_burn += 1
            else:
                iteration += 1

        # Save the groundtruth label and distribution image
        if not os.path.isdir(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        with open(self.LABEL_TXT_PATH, 'w') as outfile:
            outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(label)))
            outfile.write('# params: {0}\n'.format(self.params))

            np.savetxt(outfile, label, fmt='%-8.4f')

        np.save(self.LABEL_NPY_PATH, label)

        timex = np.arange(0, burn_in, 1)
        plt.plot(timex[0::5], probOverTime[0::5], 'o')
        plt.savefig(self.DISTRIBUTION_PNG)
        return

    def precompute_distance_matrix(self):
        np.save('/Users/taehyo/Documents/NYU/Research/DeepFDR/cHMRF/speed_test/distance.npy', np.empty((self.SIZE**3, self.SIZE**3), dtype=np.float16))

        index = 0
        xyz = np.zeros((self.SIZE**3, 3)).astype(np.float32)
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                for k in range(self.SIZE):
                    xyz[index] = np.array([i,j,k]).astype(np.float32)
                    index += 1
        # now I have the x,y,z coordinate for all roi voxels, and also the corresponding test_statistics for p voxels 
        # save this so that I can use it for mapping later 
        np.save('/Users/taehyo/Documents/NYU/Research/DeepFDR/cHMRF/speed_test/1d_roi_to_3d_mapping.npy', xyz)
        
        dist = np.memmap('/Users/taehyo/Documents/NYU/Research/DeepFDR/cHMRF/speed_test/distance.npy', dtype='float16', mode='r+', shape=(self.SIZE**3,self.SIZE**3)) # write to here

        for index in range(self.SIZE**3):
            dist[index] = d_ij(xyz[:,0], xyz[:,1], xyz[:,2], xyz[index,0], xyz[index,1], xyz[index,2]).astype(np.float16)

        dist.flush()

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
    return float64(1.0) / (float64(1.0) + (((z - k) ** 2 + (y - j) ** 2 + (x - i) ** 2)**float64(0.5)))

@vectorize([float64(float64, float64, float64, float64, float64)], target ='cpu')
def theta_ij(dist, B_1, B_1d, B_2d, B_3d):
    #return (B_1 + B_3d * dist*dist*dist + B_3r * rho*rho*rho + B_2d * dist*dist + B_2r * rho*rho + B_1d * dist + B_1r * rho)
    return B_1 + B_3d*dist*dist*dist + B_2d*dist*dist + B_1d*dist

@vectorize([float64(float64, float64)], target ='cpu')
def hj_theta(t_ij, result):
    return t_ij * result

@numba.njit(cache = True, parallel=True)
def sample(voxel_size, params, label, dist):
    result = label
    for i in numba.prange(voxel_size):
        t_ij = theta_ij(dist[i], params[1], params[2], params[3], params[4])
        theta_sum = np.sum(t_ij)

        hj_t = hj_theta(t_ij, result)
        hj_theta_sum = np.sum(hj_t)

        numerator = np.exp(-theta_sum - params[0] + hj_theta_sum)

        p = numerator / (1 + numerator)
        if np.isnan(p): #because lim inf/1+inf ~ 1
            result[i] = np.random.binomial(1, 1.0)
        else:
            result[i] = np.random.binomial(1, p)
    return result

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test_Statistics Generation')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--datapath', default='./', type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)

    numba.set_num_threads(int(numba.config.NUMBA_NUM_THREADS))
    burn_in = 1000
    num_samples = 1

    sampler = GibbsSampler(args.seed, args.datapath)

    print("NUM ITERATIONS: ", burn_in)
    print("SIZE: ", sampler.SIZE, sampler.SIZE, sampler.SIZE)

    #sampler.precompute_distance_matrix()

    sampler = GibbsSampler(args.seed, args.datapath)
    start = time.time()
    sampler.run(burn_in, num_samples)
    end = time.time()
    print("time: ", end - start)
