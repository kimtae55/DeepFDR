import torch
import torch.utils.data as Data
import os
import glob
import numpy as np
import pdb
from configure import Config
import math
import torch.nn.functional as F
import scipy.stats as stats

class DataLoader():
    #initialization
    #datapath : the data folder of bsds500
    #mode : train/test/val
    def __init__(self, mode, config, args):

        self.config = config
        
        train_input_path = os.path.join(self.config.datapath)

        if self.config.data_mode == 'single':
            train_split = 1
            valid_split = 1
        elif self.config.data_mode == 'multi':
            train_split = 5000
            valid_split = 1000

        if mode == 'train':
            self.X = np.load(train_input_path)['arr_0'][0:train_split].reshape((-1,30,30,30))
            self.y = np.load(train_input_path)['arr_0'][0:train_split].reshape((-1,30,30,30))
            self.data_len = train_split
        elif mode == 'test':
            self.X = np.load(train_input_path)['arr_0'][train_split:train_split+valid_split].reshape((-1,30,30,30))
            self.y = np.load(train_input_path)['arr_0'][train_split:train_split+valid_split].reshape((-1,30,30,30))
            self.data_len = valid_split

        self.p_value = 2*(torch.FloatTensor([1.0])-torch.FloatTensor(stats.norm.cdf(self.X.copy())))
        self.X = torch.FloatTensor(self.X) # X
        self.y = self.X.clone() # X

        # zero pad input and label by 1 on each side to make it 32x32x32
        p3d = (1, 1, 1, 1, 1, 1)
        self.X_pad = F.pad(self.X.clone(), p3d, "constant", 0) ###########check if padding is done properly
        # add dimension to match conv3d weights
        self.X_pad = self.X_pad.unsqueeze(1)
        self.y_0 = self.y.unsqueeze(1)
        self.y_1 = self.p_value.unsqueeze(1)

        #form dataset
        self.dataset = self.form_dataset()

    def form_dataset(self):
        dataset = []
        for batch_id in range(0, self.X_pad.shape[0], self.config.batch_size):
            input = self.X_pad[batch_id:min(self.X_pad.shape[0],batch_id+self.config.batch_size)]
            y_0 = self.y_0[batch_id:min(self.y_0.shape[0],batch_id+self.config.batch_size)]
            y_1 = self.y_1[batch_id:min(self.y_1.shape[0],batch_id+self.config.batch_size)]
            dataset.append(Data.TensorDataset(input, y_0, y_1))
        return Data.ConcatDataset(dataset)

    def __len__(self):
        return self.data_len // self.config.batch_size
