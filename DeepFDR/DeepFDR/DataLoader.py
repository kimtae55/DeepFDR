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
from util import qvalue

class DataLoader():
    #initialization
    #datapath : the data folder of bsds500
    #mode : train/test/val
    def __init__(self, mode, config, args):

        self.config = config
        self.config.inputsize = (30,30,30)
        
        train_input_path = os.path.join(self.config.datapath)

        if mode == 'train':
            self.X = np.load(train_input_path)[self.config.sample_number:self.config.sample_number+1].reshape((-1,)+self.config.inputsize)
            self.y = np.load(train_input_path)[self.config.sample_number:self.config.sample_number+1].reshape((-1,)+self.config.inputsize)
            self.data_len = 1
            #self.X = np.load(self.config.labelpath).reshape((-1,30,30,30))
            #self.y = np.load(self.config.labelpath).reshape((-1,30,30,30))

        # if p_value is less than 0.1, it's considered a 1 in segementation 
        # and our model tries to predict P(h=1), so I should do 1-output if I want to use MSELoss against q-value or p-value 
        self.p_value = 2.0*(1.0-stats.norm.cdf(np.fabs(self.X.copy())))
        self.q_value = qvalue(self.p_value.ravel(), threshold=0.1)[1].reshape((-1,)+self.config.inputsize)
        self.q_value = torch.FloatTensor(self.q_value)
        self.p_value = torch.FloatTensor(self.p_value)
        self.X = torch.FloatTensor(self.X) # X
        self.y = self.X.clone() # X

        # zero pad input and label by 1 on each side to make it 32x32x32
        p3d = (1, 1, 1, 1, 1, 1)
        self.X_pad = F.pad(self.X.clone(), p3d, "constant", 0) ###########check if padding is done properly
        # add dimension to match conv3d weights
        self.X_pad = self.X_pad.unsqueeze(1)
        self.y_0 = self.y.unsqueeze(1)
        self.y_1 = self.p_value.unsqueeze(1)
        self.y_2 = self.q_value.unsqueeze(1)

        #form dataset
        self.dataset = self.form_dataset()

    def form_dataset(self):
        dataset = []
        for batch_id in range(0, self.X_pad.shape[0], self.config.batch_size):
            input = self.X_pad[batch_id:min(self.X_pad.shape[0],batch_id+self.config.batch_size)]
            y_0 = self.y_0[batch_id:min(self.y_0.shape[0],batch_id+self.config.batch_size)]
            y_1 = self.y_1[batch_id:min(self.y_1.shape[0],batch_id+self.config.batch_size)]
            y_2 = self.y_2[batch_id:min(self.y_2.shape[0],batch_id+self.config.batch_size)]
            dataset.append(Data.TensorDataset(input, y_0, y_1, y_2))
        return Data.ConcatDataset(dataset)

    def __len__(self):
        return self.data_len // self.config.batch_size
