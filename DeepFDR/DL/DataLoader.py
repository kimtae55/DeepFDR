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
    def __init__(self, mode, config):

        self.config = config
        self.raw_input = []
        self.raw_target = []
        self.mode = mode

        train_input_path = self.config.datapath + mode
        train_input = os.path.join(train_input_path, '*.npy')
        input_filelist = glob.glob(train_input)

        train_target_path = self.config.datapath + mode + "_target"
        train_target = os.path.join(train_target_path, '*.npy')
        target_filelist = glob.glob(train_target)

        self.data_len = len(input_filelist)

        #load the files
        for input in input_filelist:
            self.raw_input.append(np.load(input).reshape((config.inputsize[0],config.inputsize[1],config.inputsize[2])))

        for target in target_filelist:
            self.raw_target.append(np.load(target).reshape((config.inputsize[0],config.inputsize[1],config.inputsize[2])))

        self.raw_input = np.stack(self.raw_input,axis = 0)
        self.raw_target = np.stack(self.raw_target,axis = 0)

        #form dataset
        self.dataset = self.form_dataset()

    def form_dataset(self):
        dataset = []
        for batch_id in range(0, self.raw_input.shape[0], self.config.batch_size):
            input = self.raw_input[batch_id:min(self.raw_input.shape[0],batch_id+self.config.batch_size)]
            target = self.raw_target[batch_id:min(self.raw_input.shape[0],batch_id+self.config.batch_size)]
            dataset.append(Data.TensorDataset(torch.from_numpy(input).float(), torch.from_numpy(target).float()))
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
