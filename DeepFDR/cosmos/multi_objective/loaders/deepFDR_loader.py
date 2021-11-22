import torch
import torch.utils.data as Data
import os
import glob
import numpy as np
import math
import torch.nn.functional as F
from torch.distributions import normal

class DataLoader(torch.utils.data.Dataset):
    #initialization
    #datapath : the data folder of bsds500
    #mode : train/test/val
    def __init__(self, datapath, split, **kwargs):
        train_input_path = os.path.join(datapath, 'data', 'data.npz')
        target_input_path = os.path.join(datapath, 'data', 'label.npz')

        train_split = 100
        valid_split = 100

        if split == 'train':
            self.X = np.load(train_input_path)['arr_0'][0:train_split]
            self.y = np.load(target_input_path)['arr_0'][0:train_split]
        elif split == 'test':
            self.X = np.load(train_input_path)['arr_0'][train_split:train_split+valid_split]
            self.y = np.load(target_input_path)['arr_0'][train_split:train_split+valid_split]

        self.X = torch.FloatTensor(self.X)
        self.y = float(1.0) - torch.FloatTensor(self.y) # P(h=0|x)

        # zero pad input and label by 1 on each side to make it 32x32x32
        p3d = (1, 1, 1, 1, 1, 1)
        self.X_pad = F.pad(self.X.clone(), p3d, "constant", 0) ###########check if padding is done properly
        # add dimension to match conv3d weights
        self.X_pad = self.X_pad.unsqueeze(1)
        self.y_0 = self.y.unsqueeze(1)

        # compute label for reconstruction unet (standard cdf)
        gaussian = normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.y_1 = gaussian.cdf(self.X)
        self.y_1 = self.y_1.unsqueeze(1)

        print("X: ", self.X_pad.shape)
        print("y0: ", self.y_0.shape)
        print("y1: ", self.y_1.shape)

    def __getitem__(self, index):
        return dict(data=self.X_pad[index], labels_l=self.y_0[index], labels_r=self.y_1[index])
    
    def __len__(self):
        return len(self.X)

    def task_names(self):
        return ['l', 'r']