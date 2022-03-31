import numpy as np
import os
import shutil
import fnmatch
import math
import torch
import sys
from sympy import S
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

    def loadTheta(self,path):
        self.theta = np.load(path)
        self.SIZE = self.theta.shape[0]
 
    def generate_x(self,  p_l, mu_l, sigma_l, L, subjects, groups):
        mu, sigma = 0, 1
        v_0 = np.zeros((Data.SIZE, Data.SIZE, Data.SIZE))
        v_1 = np.zeros((Data.SIZE, Data.SIZE, Data.SIZE))
        for i in range(Data.SIZE):
            for j in range(Data.SIZE):
                for k in range(Data.SIZE):
                    if self.theta[i][j][k]:
                        bern = np.random.binomial(1, p_l[1])
                        for sn in range(subjects):
                            if bern:
                                v_0[i][j][k] += np.random.normal(mu,np.sqrt(sigma))
                                v_1[i][j][k] += np.random.normal(mu_l[1] *  (math.sqrt((1.0/subjects)+(1.0/subjects))), np.sqrt(sigma_l[1]))
                            else:
                                v_0[i][j][k] += np.random.normal(mu,np.sqrt(sigma))
                                v_1[i][j][k] += np.random.normal(mu_l[0] *  (math.sqrt((1.0/subjects)+(1.0/subjects))), np.sqrt(sigma_l[0]))
                    else:
                        for sn in range(subjects):
                            v_0[i][j][k] += np.random.normal(mu,np.sqrt(sigma))
                            v_1[i][j][k] += np.random.normal(mu,np.sqrt(sigma))

        x = np.zeros((Data.SIZE, Data.SIZE, Data.SIZE))
        for i in range(Data.SIZE):
            for j in range(Data.SIZE):
                for k in range(Data.SIZE):
                    x[i][j][k] = ((1.0/subjects)*v_0[i][j][k] - (1.0/subjects)*v_1[i][j][k]) / (math.sqrt((1.0/subjects)+(1.0/subjects)))

        filename = '../data/model2/' + str(rng_seed) + '/x_val.txt'
        savepath = os.path.join(os.getcwd(),filename)
        with open(savepath, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(x.shape))
            outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(x)))
            for data_slice in x:
                np.savetxt(outfile, data_slice, fmt='%-7.6f')
                outfile.write('# New z slice\n')

    def generate_x_gaussian(self, p_l, mu_l, sigma_l2, groups):
        '''
        Gaussian mixture
        ''' 
        mu, sigma = 0, 1
        x = np.zeros(self.SIZE)

        for i in range(self.SIZE):
            if self.theta[i]:
                r = np.random.choice(groups, p=p_l)
                x[i] = np.random.normal(mu_l[r], np.sqrt(sigma_l2[r]))
            else:
                x[i] = np.random.normal(mu,sigma)

        np.save(os.path.join(self.SAVE_PATH, 'x'), x)

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
    parser.add_argument('--savepath', default='./', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--pL', nargs='*', type=float)
    parser.add_argument('--muL', nargs='*', type=float)
    parser.add_argument('--sigmaL2', nargs='*', type=float)
    parser.add_argument('--groups', default = 2, type=int)
    parser.add_argument('--distribution', default='gaussian_mixture', type=str)
    parser.add_argument('--a', default = 4.0, type=float)
    parser.add_argument('--loc', default = 1.0, type=float)
    parser.add_argument('--scale', default = 4.0, type=float)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data = Data(args.seed, args.savepath)
    data.loadTheta(os.path.join(args.savepath, str(args.seed),'label/label.npy'))

    print("RAND: ", args.seed)
    print("SIZE: {0}".format(data.SIZE))

    print("Generating Test Statistics...")
    if args.distribution == 'gaussian_mixture':
        groups = args.groups
        data.generate_x_gaussian(p_l=args.pL, mu_l=args.muL, sigma_l2=args.sigmaL2, groups=groups)
    else:
        data.generate_x_nongaussian(distribution=args.distribution, mean=args.loc, scale=args.scale, a=args.a)




