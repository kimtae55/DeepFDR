import torch
import torch.utils.data as Data
import os
import glob
import numpy as np
import pdb
from configure import Config
import math

class DataLoader():
    #initialization
    #datapath : the data folder of bsds500
    #mode : train/test/val
    def __init__(self, mode, config, args):

        self.config = config

        train_input_path = os.path.join(datapath, 'data', 'data.npz')
        target_input_path = os.path.join(datapath, 'data', 'label.npz')

        train_split = 5000
        valid_split = 1000

        if mode == 'train':
            self.X = np.load(train_input_path)['arr_0'][0:train_split]
            self.y = np.load(target_input_path)['arr_0'][0:train_split]
            self.data_len = 5000
        elif mode == 'test':
            self.X = np.load(train_input_path)['arr_0'][train_split:train_split+valid_split]
            self.y = np.load(target_input_path)['arr_0'][train_split:train_split+valid_split]
            self.data_len = 1000

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

if __name__ == "__main__":
    config = Config()

    trainset = DataLoader("train", config)
    trainloader = trainset.torch_loader()

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print(inputs, labels)
