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

    def loadNumpy(self,path):
        self.theta = np.loadtxt(path).reshape((Data.SIZE, Data.SIZE, Data.SIZE))
        return self.theta

    @staticmethod
    def saveAsNpy(array):
        return np.save('label.npy', array)

    @staticmethod
    def loadNpy(path):
        return np.load('label.npy').reshape((Data.SIZE, Data.SIZE, Data.SIZE))

    def loadTheta(self, path):
        self.theta = torch.load(path).cpu().numpy()
 
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

if __name__ == "__main__":
    rng_seed = int(sys.argv[1])
    mul_1 = float(sys.argv[2])
    mul_2 = float(sys.argv[3])
    sigma_1 = float(sys.argv[4])
    sigma_2 = float(sys.argv[5])

    print("RAND: ", rng_seed)
    np.random.seed(rng_seed)
    data = Data(rng_seed)
    theta_path = os.path.join(os.getcwd(), '../data/model2/' + str(rng_seed) +'/label/label.txt')
    data.loadNumpy(theta_path)
    p_l = np.array([0.5,0.5])
    mu_l = np.array([mul_1,mul_2])
    sigma_l = np.array([sigma_1,sigma_2])
    L = 2
    subjects = 200
    groups = 2
    print("mu_l: ", mu_l)
    print("p_l: ", p_l)
    print("sigma_l: ", sigma_l)
    print("Generating Test Statistics...")
    data.generate_x(p_l=p_l, mu_l=mu_l, sigma_l=sigma_l, L=L, subjects=subjects, groups=groups)
