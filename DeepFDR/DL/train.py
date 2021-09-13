import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import normal
from configure import Config
from model import WNet
from DataLoader import DataLoader
from early_stop import EarlyStop
import time
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.utils.data as Data

def main():
    parser = argparse.ArgumentParser(description='DeepFDR using W-NET')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--num_gpu', type=int)
    args = parser.parse_args()

    config = Config()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if args.num_gpu > 1:
        # handle multiple machine/multiple gpu training
        multi_training(args, config)
    else:
        # handle cpu & single gpu training
        single_training(config)

def single_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainLossEnc_plot = np.zeros(config.epochs)
    trainLossDec_plot = np.zeros(config.epochs)
    testLossEnc_plot = np.zeros(config.epochs)
    testLossDec_plot = np.zeros(config.epochs)
    earlyStop = EarlyStop(patience=config.patience, threshold=config.threshold)

    trainset = DataLoader("train", config)
    trainloader = Data.DataLoader(trainset.dataset, batch_size = self.config.batch_size, shuffle = self.config.shuffle, num_workers = self.config.loadThread, pin_memory = True)
    testset = DataLoader("test", config)
    testloader = Data.DataLoader(testset.dataset, batch_size = self.config.batch_size, shuffle = self.config.shuffle, num_workers = self.config.loadThread, pin_memory = True)

    global net
    net=WNet(config.num_classes)
    if torch.cuda.is_available():
        net=net.cuda()

    global criterion, gaussian
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    gaussian = normal.Normal(0.0, 1.0)

    for epoch in range(config.epochs):  # loop over the dataset multiple times
        start_time = time.time()
        train_loss_enc, train_loss_dec = train_epoch(trainloader, optimizer)
        test_loss_enc, test_loss_dec = test_epoch(testloader, optimizer)
        end_time = time.time()

        # print statistics
        trainLossEnc_plot[epoch] = train_loss_enc
        trainLossDec_plot[epoch] = train_loss_dec
        testLossEnc_plot[epoch] = test_loss_enc
        testLossDec_plot[epoch] = test_loss_dec
        print("train_l_e: %5.3f, train_l_d: %5.3f ------ test_l_e: %5.3f, test_l_d: %5.3f" % (train_loss_enc, train_loss_dec, test_loss_enc, test_loss_dec))
        print("epoch_time: %8.2f, ~time_left: %8.2f" % (end_time-start_time, (end_time-start_time)*(config.epochs - epoch)))

        # Early stop
        if earlyStop(test_loss_enc):
            print("Early stop activated.")
            break

    # save trained model
    torch.save(net.state_dict(), config.savepath)
    # plot loss
    timex = np.arange(0, config.epochs, 1)
    plt.plot(timex, trainLossEnc_plot, color='r', label='train_e')
    plt.plot(timex, trainLossDec_plot, color='m', label='train_d')
    plt.plot(timex, testLossEnc_plot, color='b', label='test_e')
    plt.plot(timex, testLossDec_plot, color='c', label='test_d')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(config.savepath_loss)

    print('Finished Training')

# Helpful documentation for single-node multi-gpu:
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
# https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py

# multi-node multi-gpu: https://leimao.github.io/blog/PyTorch-Distributed-Training/
# kill gpus: https://leimao.github.io/blog/Kill-PyTorch-Distributed-Training-Processes/
def multi_training(args, config):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # setup WNet
    net = WNet(config.num_classes)
    net.cuda(args.local_rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    net.train()

    trainSet = DataLoader("train", config)
    trainSampler = torch.utils.data.distributed.DistributedSampler(trainSet.dataset)
    trainLoader = Data.DataLoader(dataset=trainSet.dataset, sampler=trainSampler, batch_size=config.batch_size // args.num_gpu,
                              drop_last=True, num_workers=config.loadThread, pin_memory=True)

    testSet = DataLoader("test", config)
    testSampler = torch.utils.data.distributed.DistributedSampler(testSet.dataset)
    testLoader = Data.DataLoader(dataset=testSet.dataset, sampler=testSampler, batch_size=config.batch_size // args.num_gpu,
                              drop_last=True, num_workers=config.loadThread, pin_memory=True)

    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    criterion = nn.MSELoss().cuda(args.local_rank)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    gaussian = normal.Normal(0.0, 1.0)

    trainLossEnc_plot = np.zeros(config.epochs)
    trainLossDec_plot = np.zeros(config.epochs)
    testLossEnc_plot = np.zeros(config.epochs)
    testLossDec_plot = np.zeros(config.epochs)
    earlyStop = EarlyStop(patience=config.patience, threshold=config.threshold)

    # training and validation
    torch.set_grad_enabled(True)

    for epoch in range(config.epochs):  # loop over the dataset multiple times
        trainSampler.set_epoch(epoch)  # shuffle

        train_loss_enc = 0.0
        train_loss_dec = 0.0
        test_loss_enc = 0.0
        test_loss_dec = 0.0
        start_time = time.time()

        net.train()
        for batch_idx, data in enumerate(trainLoader):  
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero pad input and label by 1 on each side to make it 32x32x32
            p3d = (1, 1, 1, 1, 1, 1)
            inputs = F.pad(inputs, p3d, "constant", 0)
            labels = F.pad(labels, p3d, "constant", 0)
            # add dimension to match conv3d weights
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)

            inputs = inputs.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank)  

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            enc = net(inputs, returns='enc')
            #define loss function here
            enc_loss=criterion(enc, labels) # should i sigmoid here? 
            enc_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            dec = net(inputs, returns='dec')
            rec_loss=criterion(dec, gaussian.cdf(inputs))
            rec_loss.backward()
            optimizer.step()

            train_loss_enc += enc_loss.item()
            train_loss_dec += rec_loss.item()

        train_loss_enc = train_loss_enc / (batch_idx + 1)
        train_loss_dec = train_loss_dec / (batch_idx + 1)

        net.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(testLoader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero pad input and label by 1 on each side to make it 32x32x32
                p3d = (1, 1, 1, 1, 1, 1)
                inputs = F.pad(inputs, p3d, "constant", 0)
                labels = F.pad(labels, p3d, "constant", 0)
                # add dimension to match conv3d weights
                inputs = inputs.unsqueeze(1)
                labels = labels.unsqueeze(1)

                # convert to cuda compatible
                inputs = inputs.cuda(args.local_rank)
                labels = labels.cuda(args.local_rank)  

                enc, dec = net(inputs)
                enc_loss=criterion(enc, labels)
                rec_loss=criterion(dec, gaussian.cdf(inputs))

                test_loss_enc += enc_loss.item()
                test_loss_dec += rec_loss.item()

        test_loss_enc = test_loss_enc / (batch_idx + 1)
        test_loss_dec = test_loss_dec / (batch_idx + 1)

        end_time = time.time()

        if args.local_rank == 0:
            # print statistics
            trainLossEnc_plot[epoch] = train_loss_enc
            trainLossDec_plot[epoch] = train_loss_dec
            testLossEnc_plot[epoch] = test_loss_enc
            testLossDec_plot[epoch] = test_loss_dec
            print("train_l_e: %5.3f, train_l_d: %5.3f ------ test_l_e: %5.3f, test_l_d: %5.3f" % (train_loss_enc, train_loss_dec, test_loss_enc, test_loss_dec))
            print("#%d: epoch_time: %8.2f, ~time_left: %8.2f" % (epoch, end_time-start_time, (end_time-start_time)*(config.epochs - epoch)))

            # Early stop
            #if earlyStop(test_loss_enc):
            #    print("Early stop activated.")
            #    break

        lr_scheduler.step()

    if args.local_rank == 0:
        # save trained model
        torch.save(net.module.state_dict(), config.savepath)
        # plot loss
        timex = np.arange(0, config.epochs, 1)
        plt.plot(timex, trainLossEnc_plot, color='r', label='train_e')
        plt.plot(timex, trainLossDec_plot, color='m', label='train_d')
        plt.plot(timex, testLossEnc_plot, color='b', label='test_e')
        plt.plot(timex, testLossDec_plot, color='c', label='test_d')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(config.savepath_loss)

        print('Finished Training')


def train_epoch(trainloader, optimizer):
    global net
    net.train()

    train_loss_enc = 0.0
    train_loss_dec = 0.0
    test_loss_enc = 0.0
    test_loss_dec = 0.0
    for batch_idx, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero pad input and label by 1 on each side to make it 32x32x32
        p3d = (1, 1, 1, 1, 1, 1)
        inputs = F.pad(inputs, p3d, "constant", 0)
        labels = F.pad(labels, p3d, "constant", 0)
        # add dimension to match conv3d weights
        inputs = inputs.unsqueeze(1)
        labels = labels.unsqueeze(1)

        # convert to cuda compatible
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        enc = net(inputs, returns='enc')
        #define loss function here
        enc_loss=criterion(torch.sigmoid(enc), labels)
        enc_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        dec = net(inputs, returns='dec')
        rec_loss=criterion(dec, gaussian.cdf(inputs))
        rec_loss.backward()
        optimizer.step()

        train_loss_enc += enc_loss.item()
        train_loss_dec += rec_loss.item()

    return train_loss_enc / (batch_idx + 1), train_loss_dec / (batch_idx + 1)


def test_epoch(testloader, optimizer):
    global net
    net.eval()

    test_loss_enc = 0.0
    test_loss_dec = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero pad input and label by 1 on each side to make it 32x32x32
            p3d = (1, 1, 1, 1, 1, 1)
            inputs = F.pad(inputs, p3d, "constant", 0)
            labels = F.pad(labels, p3d, "constant", 0)
            # add dimension to match conv3d weights
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # convert to cuda compatible
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            enc, dec = net(inputs)
            enc_loss=criterion(enc, labels)
            rec_loss=criterion(dec, gaussian.cdf(inputs))

            test_loss_enc += enc_loss.item()
            test_loss_dec += rec_loss.item()

    return test_loss_enc / (batch_idx + 1), test_loss_dec / (batch_idx + 1)

if __name__ == "__main__":
    main()
