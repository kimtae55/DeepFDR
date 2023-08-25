import numpy as np
import os
import shutil
import fnmatch
import math
import torch
import sys
from scipy.stats import skewnorm 
import argparse 
import random

class Data:
    def __init__(self, rng_seed, savepath):
        self.rng_seed = rng_seed
        self.SAVE_PATH = savepath
        self.GROUP0_PATH = os.path.join(self.SAVE_PATH, '0')
        self.GROUP1_PATH = os.path.join(self.SAVE_PATH, '1')
        self.ZIP_PATH = os.path.join(self.SAVE_PATH, 'voxels.zip')
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)

    def loadTheta(self,path):
        self.theta = np.ravel(np.load(path))
        self.SIZE = self.theta.shape[0]
 
    def generate_x_bootstrap(self, p_l, mu_l, sigma_l2, groups, sample_size, subjects):
        """
        TODO: make it work for 1 <= sample size
              test on single sample size, then test on bootstrap

        CAVEAT: Currently a for loop is used to sequentially update all voxels
                np.random.normal has a bootstrapping function that you can use by specifying a size,
                but it's giving me incorrect behaviour where all the voxels are being updated 
                with the same value. So, for now this naive method will be used. 
        """
        mu, sigma = 0, 1
        v_0 = np.zeros((subjects, self.SIZE))
        v_1 = np.zeros((subjects, self.SIZE))
        for sub in range(subjects):
            for i in range(self.SIZE):
                if self.theta[i]:
                    bern = np.random.binomial(1, p_l[1])
                    if bern:
                        v_0[sub,i] = np.random.normal(mu,np.sqrt(sigma))
                        v_1[sub,i] = np.random.normal(mu_l[1], np.sqrt(sigma_l2[1]))
                    else:
                        v_0[sub,i] = np.random.normal(mu,np.sqrt(sigma))
                        v_1[sub,i] = np.random.normal(mu_l[0], np.sqrt(sigma_l2[0]))

        # Two groups of size subjects has been created
        # Use this to either create bootstrapped samples, or a single sample for convergence testing
        # Bootstrapping will use sampling with replacement
        # Compute the z-score for each bootstrapped samples
        x_npz = np.empty((sample_size, self.SIZE))
        for sample in range(sample_size):
            indexes = np.random.choice(subjects, size=subjects, replace=True)
            sum_v1 = np.sum(v_1[indexes], axis=0)
            sum_v0 = np.sum(v_0[indexes], axis=0)
            z = ((1.0/subjects)*sum_v1 - (1.0/subjects)*sum_v0) / (np.sqrt((1.0/subjects)+(1.0/subjects)))
            x_npz[sample] = np.copy(z)

        # save results
        filename = os.path.join(self.SAVE_PATH, 'data.npz')
        np.savez(filename, x_npz) 

    def generate_x_gaussian(self, p_l, mu_l, sigma_l2, groups, sample_size):
        '''
        Gaussian mixture
        '''         
        mu, sigma = 0, 1

        npz_array = np.empty((sample_size, self.SIZE))

        for sample in range(sample_size):
            x = np.zeros(self.SIZE)

            for i in range(self.SIZE):
                if self.theta[i]:
                    r = np.random.choice(groups, p=p_l)
                    x[i] = np.random.normal(mu_l[r], np.sqrt(sigma_l2[r]))
                else:
                    x[i] = np.random.normal(mu,sigma)
            npz_array[sample] = np.copy(x)
            
        filename = os.path.join(self.SAVE_PATH, 'data.npz')
        np.savez(filename, npz_array)            

    def generate_x_nongaussian(self, distribution, mean, scale, a):
        '''
        If you want to sample from a custom distribution, use Inverse Transform Sampling:
        https://en.wikipedia.org/wiki/Inverse_transform_sampling
        ''' 
        mu, sigma = 0, 1

        x = np.zeros((Data.SIZE, Data.SIZE, Data.SIZE))

        for i in range(Data.SIZE):
            for j in range(Data.SIZE):
                for k in range(Data.SIZE):
                    if self.theta[i][j][k]:
                        # Key points:
                        # - heavy right tailed distributions
                        # - mean adjusted to 1.0, or close to null distribution to exploit weakness in parametric estimation
                        # - mean and variance exists, so that smoothing can work with non-parametric estimation

                        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html
                        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html
                        if distribution == 'gumbel':
                            # mean = loc + euler_mascheroni*scale
                            euler_mascheroni = S.EulerGamma.evalf()
                            # set mean = 1.0
                            # scale = 4.0
                            loc = mean - euler_mascheroni*scale
                            x[i][j][k] = np.random.gumbel(loc=loc, scale=scale)
                        elif distribution == 'laplace':
                            # set mean = 1.0
                            # scale = 4
                            loc = mean
                            x[i][j][k] = np.random.laplace(loc=loc, scale=scale)
                        elif distribution == 'skewed_normal': 
                            # set mean = 1.0
                            loc = mean - scale*a/np.sqrt(1+a**2)*np.sqrt(2/math.pi)
                            # scale = 1
                            # a = 4 
                            x[i][j][k] = skewnorm.rvs(a, loc=loc, scale=scale)                            
                    else:
                        x[i][j][k] = np.random.normal(mu,np.sqrt(sigma))

        filename = '../data/' + str(self.rng_seed) + '/x_val_direct.txt'
        savepath = os.path.join(os.getcwd(),filename)
        with open(savepath, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(x.shape))
            outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(x)))
            for data_slice in x:
                np.savetxt(outfile, data_slice, fmt='%-7.6f')
                outfile.write('# New z slice\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test_Statistics Generation')
    parser.add_argument('--savepath', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--pL', nargs='*', type=float)
    parser.add_argument('--muL', nargs='*', type=float)
    parser.add_argument('--sigmaL2', nargs='*', type=float)
    parser.add_argument('--groups', default = 2, type=int)
    parser.add_argument('--distribution', default='gaussian_mixture', type=str)
    parser.add_argument('--a', default = 4.0, type=float)
    parser.add_argument('--loc', default = 1.0, type=float)
    parser.add_argument('--scale', default = 4.0, type=float)
    parser.add_argument('--sample_size', default = 1, type=int)
    parser.add_argument('--subjects', default = 200, type=int)
    parser.add_argument('--labelpath', type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data = Data(args.seed, args.savepath)
    data.loadTheta(args.labelpath)

    print("RAND: ", args.seed)
    print("SIZE: {0}".format(data.SIZE))

    print("Generating Test Statistics...")
    if args.distribution == 'gaussian_mixture':
        groups = args.groups
        data.generate_x_gaussian(p_l=args.pL, mu_l=args.muL, sigma_l2=args.sigmaL2, groups=args.groups, sample_size=args.sample_size)
    elif args.distribution == 'gaussian_bootstrap':
        data.generate_x_bootstrap(p_l=args.pL, mu_l=args.muL, sigma_l2=args.sigmaL2, groups=args.groups, sample_size=args.sample_size, subjects=args.subjects)
    else:
        data.generate_x_nongaussian(distribution=args.distribution, mean=args.loc, scale=args.scale, a=args.a)




