import numpy as np
import os
import shutil
import fnmatch
import math
import torch
import sys
import h5py
import time

class Data:
    SIZE = 30

    def __init__(self, rng_seed):
        self.RHO_PATH = os.path.join(os.getcwd(), 'rho_ij.hdf5')
        self.v = torch.empty((Data.SIZE, Data.SIZE, Data.SIZE)).cuda()
        self.theta = torch.empty((Data.SIZE, Data.SIZE, Data.SIZE)).cuda()
        self.rng_seed = rng_seed
        self.file_ext = '.pt'
        self.SAVE_PATH = os.path.join(os.getcwd(), '../data/model2/' + str(rng_seed) + '/voxels')
        self.GROUP0_PATH = os.path.join(self.SAVE_PATH, '0')
        self.GROUP1_PATH = os.path.join(self.SAVE_PATH, '1')
        self.ZIP_PATH = os.path.join(self.SAVE_PATH, 'voxels.zip')
        self.bootstrap_size = 5000

    @staticmethod
    def loadNumpy(path):
        return torch.from_numpy(np.loadtxt(path).reshape((Data.SIZE, Data.SIZE, Data.SIZE)))

    @staticmethod
    def load(path):
        return torch.load(path).cuda()

    def generate_v(self, p_l, mu_l, sigma_l, L, subjects, groups):
        mu, sigma = 0.0, 1.0

        for gn in range(groups):
            directory = os.path.join(self.SAVE_PATH, str(gn) + '/')
            if not os.path.exists(directory):
                os.makedirs(directory)
            # compute voxels
            for sn in range(subjects):
                prob = torch.rand(1)
                if prob < p_l[0]:
                    mixture = torch.normal(mean=float(gn * mu_l[0]), std=float(torch.sqrt(torch.tensor([sigma_l[0]]))), size=(Data.SIZE, Data.SIZE, Data.SIZE)).cuda()
                else:
                    mixture = torch.normal(mean= float(gn * mu_l[1]), std=float(torch.sqrt(torch.tensor([sigma_l[1]]))), size=(Data.SIZE, Data.SIZE, Data.SIZE)).cuda()

                self.v = (1-self.theta)*torch.normal(mean=float(mu), std=float(torch.sqrt(torch.tensor([sigma]))), size=(Data.SIZE, Data.SIZE, Data.SIZE)).cuda() + self.theta*mixture

                filename = 'k=' + str(gn) + '_j=' + str(sn) + self.file_ext
                savepath = os.path.join(directory, filename)
                torch.save(self.v, savepath)

        shutil.make_archive(self.SAVE_PATH, 'zip', self.SAVE_PATH)

    def generate_x(self, subjects):
        # generate x
        # load data from zipped folder
        # use simulation model (1) to generate x statistics
        # save into txt
        g1 = torch.empty((subjects, Data.SIZE, Data.SIZE, Data.SIZE)).cuda()
        g2 = torch.empty((subjects, Data.SIZE, Data.SIZE, Data.SIZE)).cuda()
        if not os.path.exists(self.SAVE_PATH) and os.path.exists(self.ZIP_PATH):
            shutil.unpack_archive(self.ZIP_PATH, self.SAVE_PATH, 'zip')

        v_0 = torch.zeros((Data.SIZE, Data.SIZE, Data.SIZE)).cuda()
        for i, filename in enumerate(os.listdir(self.GROUP0_PATH)):
            if filename.endswith(self.file_ext):
                tmp = self.load(os.path.join(self.GROUP0_PATH,filename))
                v_0 = v_0 + tmp
                g1[i] = tmp

        v_1 = torch.zeros((Data.SIZE, Data.SIZE, Data.SIZE)).cuda()
        for i, filename in enumerate(os.listdir(self.GROUP1_PATH)):
            if filename.endswith(self.file_ext):
                tmp = self.load(os.path.join(self.GROUP1_PATH,filename))
                v_1 = v_1 + tmp
                g2[i] = tmp

        n0 = len(fnmatch.filter(os.listdir(self.GROUP0_PATH), '*' + self.file_ext))
        n1 = len(fnmatch.filter(os.listdir(self.GROUP1_PATH), '*' + self.file_ext))

        x = (((1.0 / n0) * v_0 - (1.0 / n1) * v_1) / (math.sqrt((1.0 / n0) + (1.0 / n1)))).cuda()


        filename = '../data/model2/' + str(rng_seed) + '/x' + self.file_ext
        savepath = os.path.join(os.getcwd(), filename)
        torch.save(x, savepath)

        filename = '../data/model2/' + str(rng_seed) + '/x_val.txt'
        savepath = os.path.join(os.getcwd(),filename)
        with open(savepath, 'w') as outfile:
            for data_slice in x.detach().cpu().numpy():
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
                outfile.write('# New z slice\n')

        '''
        # generate correlation using boostrapping
        x_bootstrap = torch.empty((self.bootstrap_size ,Data.SIZE, Data.SIZE, Data.SIZE)).cuda()

        for i in range(self.bootstrap_size):
            sample_g1 = g1[torch.randint(n0, (n0,))].cuda()
            sample_g2 = g2[torch.randint(n1, (n1,))].cuda()
            x_bootstrap[i] = ((1.0 / n0) * torch.sum(sample_g1, dim=0) - (1.0 / n1) * torch.sum(sample_g2, dim=0)) / (math.sqrt((1.0 / n0) + (1.0 / n1)))

        x_mean = torch.sum(x_bootstrap, dim=0) / self.bootstrap_size

        x_bootstrap = x_bootstrap.detach().cpu().numpy()
        x_mean = x_mean.detach().cpu().numpy()
        start = time.time()

        with h5py.File(self.RHO_PATH, 'w') as f:
            for i in range(Data.SIZE):
                for j in range(Data.SIZE):
                    for k in range(Data.SIZE):
                        # function for d_ij
                        # function for rho_ij\
                        arr = np.zeros((Data.SIZE, Data.SIZE, Data.SIZE))
                        for j_i in range(Data.SIZE):
                            for j_j in range(Data.SIZE):
                                for j_k in range(Data.SIZE):
                                    rho_ij = np.sum((x_bootstrap[:,i,j,k] - x_mean[i,j,k])*(x_bootstrap[:,j_i,j_j,j_k] - x_mean[j_i,j_j,j_k])) / self.bootstrap_size
                                    arr[j_i][j_j][j_k] = rho_ij
                        # save to file
                        f.create_dataset(str(i) + "_" + str(j) + "_" + str(k), data=arr)
                        print(i*Data.SIZE*Data.SIZE + j*Data.SIZE + k, ": ", time.time() - start, "s")
        '''
        return

if __name__ == "__main__":
    rng_seed = sys.argv[1]
    print("RAND: ", rng_seed)
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    print("GPU: ", torch.cuda.get_device_name(0))

    data = Data(rng_seed)
    theta_path = os.path.join(os.getcwd(), '../data/model2/' + str(rng_seed) + '/label/label.txt')
    data.loadNumpy(theta_path)
    p_l = torch.tensor([0.5, 0.5])
    mu_l = torch.tensor([-1.0, 3.0])
    sigma_l = torch.tensor([1.0, 1.0])
    L = 2
    subjects = 200
    groups = 2
    print("Generating voxels...")
    print("mu_l: ", mu_l)
    print("p_l: ", p_l)
    print("sigma_l: ", sigma_l)
    data.generate_v(p_l=p_l, mu_l=mu_l, sigma_l=sigma_l, L=L, subjects=subjects, groups=groups)
    # mu_1 = -2 mu_2 = 2
    print("Generating Test Statistics...")
    data.generate_x(subjects=200)


