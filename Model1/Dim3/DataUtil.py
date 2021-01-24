import numpy as np
import os
import shutil
import fnmatch
import math
import matplotlib.pyplot as plt


class Data:
    VOXEL_SIZE = 15
    SAVE_PATH = os.path.join(os.getcwd(), '../data_15x15x15/voxels')
    GROUP0_PATH = os.path.join(os.getcwd(), '../data_15x15x15/voxels/0/')
    GROUP1_PATH = os.path.join(os.getcwd(), '../data_15x15x15/voxels/1/')
    ZIP_PATH = os.path.join(os.getcwd(), '../data_15x15x15/voxels.zip')

    def __init__(self):
        self.v = np.empty((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        self.theta = np.empty((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))

    @staticmethod
    def loadX(path):
        x = np.loadtxt(path).reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        return x

    def loadTheta(self, path):
        self.theta = np.loadtxt(path).reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))

    def generate_v(self, p_l, mu_l, sigma_l, L, subjects, groups):
        # use equation (4) to generate n voxels for j subjects from k groups
        # save the voxels in .txt into k folders
        # will generate voxels.zip
        mu, sigma = 0, 1

        for gn in range(groups):
            directory = os.path.join(os.getcwd(), '../data_15x15x15/voxels/' + str(gn) + '/')
            if not os.path.exists(directory):
                os.makedirs(directory)
            # compute voxels
            for sn in range(subjects):
                for i in range(Data.VOXEL_SIZE):
                    for j in range(Data.VOXEL_SIZE):
                        for k in range(Data.VOXEL_SIZE):
                            sum = 0
                            for l in range(L):
                                sum = sum + p_l*np.random.normal(gn*mu_l,sigma_l**2)
                            self.v[i][j][k] = (1-self.theta[i][j][k])*np.random.normal(mu,sigma) \
                                              + self.theta[i][j][k]*sum
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
        shutil.make_archive(Data.SAVE_PATH, 'zip', Data.SAVE_PATH)

    def generate_x(self, subjects):
        # load data from zipped folder
        # use simulation model (1) to generate x statistics
        # save into txt
        if not os.path.exists(Data.SAVE_PATH) and os.path.exists(Data.ZIP_PATH):
            shutil.unpack_archive(Data.ZIP_PATH, Data.SAVE_PATH, 'zip')

        v_0 = np.zeros((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        for i, filename in enumerate(os.listdir(Data.GROUP0_PATH)):
            if filename.endswith(".txt"):
                v_0 = v_0 + np.loadtxt(os.path.join(Data.GROUP0_PATH,filename)).reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))

        v_1 = np.zeros((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        for i, filename in enumerate(os.listdir(Data.GROUP1_PATH)):
            if filename.endswith(".txt"):
                v_1 = v_1 + np.loadtxt(os.path.join(Data.GROUP1_PATH,filename)).reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))

        x = np.zeros((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        n0 = len(fnmatch.filter(os.listdir(Data.GROUP0_PATH), '*.txt'))
        n1 = len(fnmatch.filter(os.listdir(Data.GROUP0_PATH), '*.txt'))
        for i in range(Data.VOXEL_SIZE):
            for j in range(Data.VOXEL_SIZE):
                for k in range(Data.VOXEL_SIZE):
                    x[i][j][k] = ((1/n0)*v_0[i][j][k] - (1/n1)*v_1[i][j][k]) / (math.sqrt((1/n0)+(1/n1)))

        filename = '../data_15x15x15/x_val.txt'
        savepath = os.path.join(os.getcwd(),filename)
        with open(savepath, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(x.shape))
            outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(x)))
            for data_slice in x:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
                outfile.write('# New z slice\n')



if __name__ == "__main__":
    np.random.seed(12345)
    data = Data()
    theta_path = os.path.join(os.getcwd(), '../data_15x15x15/beta_02_iter_3000.txt')
    data.loadTheta(theta_path)
    data.generate_v(p_l=0.5, mu_l=0.5, sigma_l=1, L=2, subjects=200, groups=2)
    data.generate_x(subjects=200)

    x = np.loadtxt(os.path.join('../data_15x15x15/x_val.txt')).reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
    for k in range(Data.VOXEL_SIZE):
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(x[k])
        plt.title('x_statistic')
        plt.show()

