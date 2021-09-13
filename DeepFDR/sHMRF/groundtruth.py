import numpy as np
import os
import math
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_statistic import Data
import sys

class GibbsSampler:
    def __init__(self, rng_seed):
        self.white_map = torch.from_numpy(np.indices((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE)).sum(axis=0) % 2)
        self.white_map = self.white_map.cuda()
        ones = torch.ones((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE)).cuda()
        self.black_map = torch.logical_xor(self.white_map, ones).type(torch.FloatTensor)
        self.black_map = self.black_map.cuda()
        
        # RUN FROM INSIDE DIM3 FOLDER
        self.KERNEL_PATH = os.path.join(os.getcwd(), 'kernel.txt')
        self.SAVE_DIR = os.path.join(os.getcwd(), '../../data/model1/' + str(rng_seed) +'/label')
        self.LABEL_PATH = os.path.join(self.SAVE_DIR, 'label.pt')
        self.LABEL_TXT_PATH = os.path.join(self.SAVE_DIR, 'label.txt')
        self.DISTRIBUTION_PNG = os.path.join(self.SAVE_DIR, 'distribution.png')
        self.LOG_PATH = os.path.join(self.SAVE_DIR, 'log.txt')
        return

    def run(self, burn_in, n_iter):
        # Store P(theta=1) over time
        probOverTime = torch.tensor(np.zeros(burn_in))

        # initialize labels
        init_prob = torch.full((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE), 0.5)
        label = torch.bernoulli(init_prob).cuda()  # p = 0.5 to choose 1
        kernel = torch.from_numpy(np.loadtxt(self.KERNEL_PATH).reshape((3,3,3))).float()
        kernel = torch.unsqueeze(kernel, 0)
        kernel = torch.unsqueeze(kernel, 0)
        kernel = kernel.cuda()

        iteration = 0
        iter_burn = 0
        while iteration < n_iter:
            # calculate sum_nn
            # sum of neighboring pixels is a convolution operation
            label = torch.unsqueeze(label, 0)
            label = torch.unsqueeze(label, 0)
            sum_nn = torch.nn.functional.conv3d(label, kernel, stride=1, padding=1).squeeze()

            # calculate exp_sum_nn
            beta = 0.8
            h = -2.5
            numerator = torch.exp(sum_nn*beta + h)
            conditional_distribution = numerator / (1 + numerator)
            # then put it into bernoulli
            white = torch.bernoulli(conditional_distribution)
            white = torch.logical_and(white, self.white_map).type(torch.FloatTensor).cuda()
            #print(white)
            # save a pre_label 
            # keep black tiles, replace white tiles with pre_label 
            # compute sum_nn again for white tiles now
            # put it into bernoulli, get white_labels
            # now merge the white label with black label 
            # (how to merge using gpu?) --> logical_or
            # calculate sum_nn
            # sum of neighboring pixels is a convolution operation
            # we have updated the white labels, so merge it with label
            black_label = torch.logical_and(label.squeeze(), self.black_map).type(torch.FloatTensor).cuda()
            label = torch.logical_or(black_label, white).type(torch.FloatTensor).cuda()
            label = torch.unsqueeze(label, 0)
            label = torch.unsqueeze(label, 0)
            sum_nn = torch.nn.functional.conv3d(label, kernel, stride=1, padding=1).squeeze()
            
            # calculate exp_sum_nn
            numerator = torch.exp(sum_nn*beta + h)
            conditional_distribution = numerator / (1 + numerator)
            # then put it into bernoulli
            black = torch.bernoulli(conditional_distribution)
            black = torch.logical_and(black, self.black_map).type(torch.FloatTensor).cuda()
            label = torch.logical_or(black, white).type(torch.FloatTensor).cuda()
            
            if iter_burn < burn_in:
                probOverTime[iter_burn] = torch.count_nonzero(label) / (Data.VOXEL_SIZE * Data.VOXEL_SIZE * Data.VOXEL_SIZE)
                iter_burn = iter_burn + 1
                #print("burn in: ", iter_burn)
            else:
                iteration += 1
        if os.path.isdir(self.SAVE_DIR):
            torch.save(label, self.LABEL_PATH)
            # Save final signal_file
            with open(self.LABEL_TXT_PATH, 'w') as outfile:
                outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(label.cpu().numpy())))
                for data_slice in label.cpu().numpy():
                    np.savetxt(outfile, data_slice, fmt='%-8.4f')
                    outfile.write('# New z slice\n')
        else:
            os.makedirs(self.SAVE_DIR)
            torch.save(label, self.LABEL_PATH)
            # Save final signal_file
            with open(self.LABEL_TXT_PATH, 'w') as outfile:
                outfile.write('# H = 1: {0}\n'.format(np.count_nonzero(label.cpu().numpy())))
                for data_slice in label.cpu().numpy():
                    np.savetxt(outfile, data_slice, fmt='%-8.4f')
                    outfile.write('# New z slice\n')
            

        print(probOverTime.shape)
        timex = torch.arange(0, burn_in, 1)
        plt.plot(timex[0::5], probOverTime[0::5], 'o')
        plt.savefig(self.DISTRIBUTION_PNG)

        return

if __name__ == "__main__":
    rng_seed = sys.argv[1]
    print("RAND: ", rng_seed)
    start = time.time()

    torch.manual_seed(rng_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    burn_in = 1000
    n_iter = 1

    print("GPU: ", torch.cuda.get_device_name(0))
    print("NUM ITERATIONS: ", burn_in)
    print("VOXEL_SIZE: ", Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE)

    start = time.time()
    sampler = GibbsSampler(rng_seed)
    sampler.run(burn_in, n_iter)
    end = time.time()
    print("Time Elapsed:", end - start)



    
    
    
    
