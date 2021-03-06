import numpy as np
import os
import shutil
import fnmatch
import math
import matplotlib.pyplot as plt
import torch
import sys

class Data:
    VOXEL_SIZE = 15

    def __init__(self, rng_seed):
        self.v = np.empty((Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        self.theta = np.empty((Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        self.rng_seed = rng_seed
        self.SAVE_PATH = os.path.join(os.getcwd(), '../data/' + str(rng_seed) + '/voxels')
        self.GROUP0_PATH = os.path.join(self.SAVE_PATH, '0')
        self.GROUP1_PATH = os.path.join(self.SAVE_PATH, '1')
        self.ZIP_PATH = os.path.join(self.SAVE_PATH, 'voxels.zip')

    @staticmethod
    def load(path):
        x = np.loadtxt(path).reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        return x

    def loadTheta(self, path):
        self.theta = torch.load(path).cpu().numpy()

    def generate_v(self, p_l, mu_l, sigma_l, L, subjects, groups):
        # use equation (4) to generate n voxels for j subjects from k groups
        # save the voxels in .txt into k folders
        # will generate voxels.zip
        mu, sigma = 0, 1

        for gn in range(groups):
            directory = os.path.join(self.SAVE_PATH, str(gn) + '/')
            if not os.path.exists(directory):
                os.makedirs(directory)
            # compute voxels
            for sn in range(subjects):
                for i in range(Data.VOXEL_SIZE):
                    for j in range(Data.VOXEL_SIZE):
                        sum = 0
                        for l in range(L):
                            sum = sum + p_l[l]*np.random.normal(gn*mu_l[l],np.sqrt(sigma_l[l]))
                        self.v[i][j] = (1-self.theta[i][j])*np.random.normal(mu,np.sqrt(sigma)) \
                                          + self.theta[i][j]*sum
                # save datum to correct folder
                filename = 'k=' + str(gn) + '_j=' + str(sn) + '.txt'
                savepath = os.path.join(directory, filename)
                with open(savepath, 'w') as outfile:
                    outfile.write('# Array shape: {0}\n'.format(self.v.shape))
                    outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(self.v)))
                    for data_slice in self.v:
                        np.savetxt(outfile, data_slice, fmt='%-7.2f')
                        outfile.write('# New z slice\n')
        # zip folder
        shutil.make_archive(self.SAVE_PATH, 'zip', self.SAVE_PATH)

    def generate_x(self, subjects):
        # load data from zipped folder
        # use simulation model (1) to generate x statistics
        # save into txt
        if not os.path.exists(self.SAVE_PATH) and os.path.exists(self.ZIP_PATH):
            shutil.unpack_archive(self.ZIP_PATH, self.SAVE_PATH, 'zip')

        v_0 = np.zeros((Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        for i, filename in enumerate(os.listdir(self.GROUP0_PATH)):
            if filename.endswith(".txt"):
                v_0 = v_0 + np.loadtxt(os.path.join(self.GROUP0_PATH,filename)).reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE))

        v_1 = np.zeros((Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        for i, filename in enumerate(os.listdir(self.GROUP1_PATH)):
            if filename.endswith(".txt"):
                v_1 = v_1 + np.loadtxt(os.path.join(self.GROUP1_PATH,filename)).reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE))

        x = np.zeros((Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        n0 = len(fnmatch.filter(os.listdir(self.GROUP0_PATH), '*.txt'))
        n1 = len(fnmatch.filter(os.listdir(self.GROUP1_PATH), '*.txt'))
        for i in range(Data.VOXEL_SIZE):
            for j in range(Data.VOXEL_SIZE):
                x[i][j] = ((1/n0)*v_0[i][j] - (1/n1)*v_1[i][j]) / (math.sqrt((1/n0)+(1/n1)))

        filename = '../data/' + str(rng_seed) + '/x_val.txt'
        savepath = os.path.join(os.getcwd(),filename)
        with open(savepath, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(x.shape))
            outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(x)))
            for data_slice in x:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
                outfile.write('# New z slice\n')



if __name__ == "__main__":
    rng_seed = int(sys.argv[1])
    print("RAND: ", rng_seed)
    np.random.seed(rng_seed)
    data = Data(rng_seed)
    theta_path = os.path.join(os.getcwd(), '../data/' + str(rng_seed) +'/label/label.pt')
    data.loadTheta(theta_path)
    p_l = np.array([0.5,0.5]);
    mu_l = np.array([0.5,0.5]);
    sigma_l = np.array([1.0,1.0]);
    L = 2
    subjects = 200
    groups = 2
    data.generate_v(p_l=p_l, mu_l=mu_l, sigma_l=sigma_l, L=L, subjects=subjects, groups=groups)
    # mu_1 = -2 mu_2 = 2
    data.generate_x(subjects=200)

'''
    x = np.loadtxt(os.path.join('../data/x_valtest.txt')).reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
    for k in range(Data.VOXEL_SIZE):
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(x[k])
        plt.title('x_statistic')
        plt.show()
'''
