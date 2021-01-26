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
from DataUtil import Data

class GibbsSampler(object):
    # RUN FROM INSIDE DIM3 FOLDER
    KERNEL_PATH = os.path.join(os.getcwd(), 'kernel.txt')
    SAVE_DIR = os.path.join(os.getcwd(), '../data/label')
    LABEL_PATH = os.path.join(os.getcwd(), '../data/label/label.pt')
    DISTRIBUTION_PNG = os.path.join(os.getcwd(), '../data/label/distribution.png')
    LOG_PATH = os.path.join(os.getcwd(), '../data/label/log.txt')
    
    def __init__(self):
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
            
            label = torch.bernoulli(conditional_distribution)

            if iter_burn < burn_in:
                probOverTime[iter_burn] = torch.count_nonzero(label) / (Data.VOXEL_SIZE * Data.VOXEL_SIZE * Data.VOXEL_SIZE)
                iter_burn = iter_burn + 1
                #print("burn in: ", iter_burn)
            else:
                iteration += 1
        if os.path.isdir(self.SAVE_DIR):
            torch.save(label, self.LABEL_PATH)
        else:
            os.makedirs(self.SAVE_DIR)
            torch.save(label, self.LABEL_PATH)

        print(probOverTime.shape)
        timex = torch.arange(0, burn_in, 1)
        plt.plot(timex[0::5], probOverTime[0::5], 'o')
        plt.savefig(self.DISTRIBUTION_PNG)

        return

if __name__ == "__main__":
    start = time.time()

    torch.manual_seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    burn_in = 1000
    n_iter = 1

    print("GPU: ", torch.cuda.get_device_name(0))
    print("NUM ITERATIONS: ", burn_in)
    print("VOXEL_SIZE: ", Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE)

    sampler = GibbsSampler()


    sampler.run(burn_in, n_iter)

    end = time.time()
    print("Time Elapsed:", end - start)
    
    with open(sampler.LOG_PATH, 'w') as outfile:
        outfile.write('GPU: {0}\n'.format(torch.cuda.get_device_name(0)))
        outfile.write('BURN INs: {0}\n'.format(burn_in))
        outfile.write('VOXEL_SIZE: {0} {1} {2}\n'.format(Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        outfile.write('EXECUTION_TIME: {0} (s)\n'.format( end - start))

    ''' 
    # For extra visualization
    initdata = torch.load('testin.pt').cpu().numpy()
    data = torch.load('testout.pt').cpu().numpy()
    x, y, z = data.nonzero()
    df = pd.DataFrame(np.hstack((x[:, None], y[:, None], z[:, None])),columns=['z','x','y'])
    fig = px.scatter_3d(df,x = 'z',y = 'x',z='y', opacity=0.8)
    fig.show()

    for k in range(Data.VOXEL_SIZE):
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(initdata[k])
        plt.title('initlabel')
        plt.subplot(1, 2, 2)
        plt.imshow(data[k])
        plt.title('label')
        plt.show()
    
    '''
    
    
    
    
    