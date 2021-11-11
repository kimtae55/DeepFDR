import numpy as np
import os
import shutil
import fnmatch
import math
import matplotlib.pyplot as plt
import torch
import sys

class Data:
    SIZE = 30

    def __init__(self, rng_seed):
        self.v = np.empty((Data.SIZE, Data.SIZE, Data.SIZE))
        self.theta = np.empty((Data.SIZE, Data.SIZE, Data.SIZE))
        self.rng_seed = rng_seed
        self.SAVE_PATH = os.path.join(os.getcwd(), '../data/model2/' + str(rng_seed) + '/voxels')
        self.GROUP0_PATH = os.path.join(self.SAVE_PATH, '0')
        self.GROUP1_PATH = os.path.join(self.SAVE_PATH, '1')
        self.ZIP_PATH = os.path.join(self.SAVE_PATH, 'voxels.zip')

    def loadTheta(self,path):
        self.theta = np.loadtxt(path).reshape((Data.SIZE, Data.SIZE, Data.SIZE))
        return self.theta

    def generate_x_directly(self, p_l, mu_l, sigma_l, L, subjects, groups):
        mu, sigma = 0, 1
        x = np.zeros((Data.SIZE, Data.SIZE, Data.SIZE))

        for i in range(Data.SIZE):
            for j in range(Data.SIZE):
                for k in range(Data.SIZE):
                    if self.theta[i][j][k]:
                        bern = np.random.binomial(1, p_l[1])
                        if bern:
                            x[i][j][k] += np.random.normal(mu_l[1], np.sqrt(sigma_l[1]))
                        else:
                            x[i][j][k] += np.random.normal(mu_l[0], np.sqrt(sigma_l[0]))
                    else:
                        x[i][j][k] += np.random.normal(mu,np.sqrt(sigma))

        filename = '../data/model2/' + str(rng_seed) + '/x.txt'
        savepath = os.path.join(os.getcwd(),filename)
        with open(savepath, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(x.shape))
            outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(x)))
            for data_slice in x:
                np.savetxt(outfile, data_slice, fmt='%-7.6f')
                outfile.write('# New z slice\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test_Statistics Generation')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--p_0', default = 0.5, type=float)
    parser.add_argument('--p_1', default = 0.5, type=float)
    parser.add_argument('--mu_0', default = -2.0, type=float)
    parser.add_argument('--mu_1', default = 2.0, type=float)
    parser.add_argument('--sig_0', default = 1.0, type=float)
    parser.add_argument('--sig_1', default = 1.0, type=float)  
    parser.add_argument('--groups', default=2, type=int)
    parser.add_argument('--subjects', default=200, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data = Data(args.seed)
    data.loadTheta('../data/model2/' + str(rng_seed) + '/label/label.txt')
    p_l = np.array([args.p_0,args.p_1])
    mu_l = np.array([args.mu_0,args.mu_1])
    sigma_l = np.array([args.sig_0,args.sig_1])
    subjects = args.subjects
    groups = args.groups
    sample_size = args.sample_size
    print("mu_l: ", mu_l)
    print("p_l: ", p_l)
    print("sigma_l: ", sigma_l)
    print("Generating Test Statistics...")

    data.generate_x_directly(p_l=p_l, mu_l=mu_l, sigma_l=sigma_l, L=L, subjects=subjects, groups=groups)