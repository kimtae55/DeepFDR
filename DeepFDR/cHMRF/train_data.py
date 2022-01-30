import numpy as np
import os
import shutil
import fnmatch
import math
import sys
import argparse

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
        self.theta = np.load(path).reshape((Data.SIZE, Data.SIZE, Data.SIZE))

    def loadTheta(self, path):
        self.theta = torch.load(path).cpu().numpy()

    def bootstrap_for_dl(self,  p_l, mu_l, sigma_l, L, subjects, groups):
        mu, sigma = 0, 1
        v_0 = np.zeros((subjects,Data.SIZE, Data.SIZE, Data.SIZE))
        v_1 = np.zeros((subjects,Data.SIZE, Data.SIZE, Data.SIZE))
        for i in range(Data.SIZE):
            for j in range(Data.SIZE):
                for k in range(Data.SIZE):
                    if self.theta[i][j][k]:
                        bern = np.random.binomial(1, p_l[1])
                        for sn in range(subjects):
                            if bern:
                                v_0[sn][i][j][k] = np.random.normal(mu,np.sqrt(sigma))
                                v_1[sn][i][j][k] = np.random.normal(mu_l[1] *  (math.sqrt((1.0/subjects)+(1.0/subjects))), np.sqrt(sigma_l[1]))
                            else:
                                v_0[sn][i][j][k] = np.random.normal(mu,np.sqrt(sigma))
                                v_1[sn][i][j][k] = np.random.normal(mu_l[0] *  (math.sqrt((1.0/subjects)+(1.0/subjects))), np.sqrt(sigma_l[0]))
                    else:
                        for sn in range(subjects):
                            v_0[sn][i][j][k] = np.random.normal(mu,np.sqrt(sigma))
                            v_1[sn][i][j][k] = np.random.normal(mu,np.sqrt(sigma))
        # v_0 contains 200 voxels
        # v_1 contians 200 voxels
        # Do sample with replacement for both groups
        sample_size = 200 # with replacement
        trainset_size = 5000
        testset_size = 1000
        train_directory = '../data/model2/' + str(self.rng_seed) + '/train/'
        test_directory = '../data/model2/' + str(self.rng_seed) + '/test/'

        if not os.path.isdir(train_directory):
            os.makedirs(train_directory)
        if not os.path.isdir(test_directory):
            os.makedirs(test_directory)

        for i in range(trainset_size):
            random_indices = np.random.choice(sample_size, size=sample_size, replace=True)
            samples_v0 = v_0[random_indices, :]
            samples_v1 = v_1[random_indices, :]
            filename = train_directory + str(i) + '.npy'
            x = ((1.0/subjects)*np.sum(samples_v0, axis=0) - (1.0/subjects)*np.sum(samples_v1, axis=0)) / (math.sqrt((1.0/subjects)+(1.0/subjects)))
            np.save(filename, x)

        for i in range(testset_size):
            random_indices = np.random.choice(sample_size, size=sample_size, replace=True)
            samples_v0 = v_0[random_indices, :]
            samples_v1 = v_1[random_indices, :]
            filename = test_directory + str(i) + '.npy'
            x = ((1.0/subjects)*np.sum(samples_v0, axis=0) - (1.0/subjects)*np.sum(samples_v1, axis=0)) / (math.sqrt((1.0/subjects)+(1.0/subjects)))
            np.save(filename, x)

    def generate_x(self,  p_l, mu_l, sigma_l, L, subjects, groups, sample_size=300):
        x_directory = '../data/wnet/' + str(self.rng_seed) + '/test_statistics/'
        if not os.path.isdir(x_directory):
            os.makedirs(x_directory)

        for sample in range(sample_size):
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

            filename = x_directory + str(sample) +'.npy'
            np.save(filename, x)

    def generate_x_directly(self, p_l, mu_l, sigma_l, sample_size, out_path):
        x_directory = os.path.join(out_path, 'data')
       
        if not os.path.isdir(x_directory):
            os.makedirs(x_directory)

        mu, sigma = 0, 1

        npz_array = np.empty((sample_size, 30, 30, 30))

        for sample in range(sample_size):
            x = np.zeros((Data.SIZE, Data.SIZE, Data.SIZE))
            for i in range(Data.SIZE):
                for j in range(Data.SIZE):
                    for k in range(Data.SIZE):
                        if self.theta[i][j][k]:
                            bern = np.random.binomial(1, p_l[1])
                            if bern:
                                x[i][j][k] = np.random.normal(mu_l[1], np.sqrt(sigma_l[1]))
                            else:
                                x[i][j][k] = np.random.normal(mu_l[0], np.sqrt(sigma_l[0]))
                        else:
                            x[i][j][k] = np.random.normal(mu,np.sqrt(sigma))
            npz_array[sample] = np.copy(x)
        filename = os.path.join(x_directory,  'data.npz')
        np.savez(filename, npz_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test_Statistics Generation')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--p_0', default = 0.5, type=float)
    parser.add_argument('--p_1', default = 0.5, type=float)
    parser.add_argument('--mu_0', type=float)
    parser.add_argument('--mu_1', type=float)
    parser.add_argument('--sig_0', type=float)
    parser.add_argument('--sig_1', type=float)  
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--groups', default=2, type=int)
    parser.add_argument('--subjects', default=200, type=int)
    parser.add_argument('--sample_size', default=6000, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    data = Data(args.seed)
    data.loadNumpy(args.label_path)
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

    data.generate_x_directly(p_l=p_l, mu_l=mu_l, sigma_l=sigma_l, sample_size=sample_size, out_path=args.out_path)
    