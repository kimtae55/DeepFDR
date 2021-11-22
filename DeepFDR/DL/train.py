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
from torchinfo import summary

def main():
    parser = argparse.ArgumentParser(description='DeepFDR using W-NET')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--num_gpu', default=1, type=int)
    parser.add_argument('--k_folds', default=6, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--datapath', type=str)
    args = parser.parse_args()

    config = Config()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    if args.num_gpu > 1:
        # handle multiple machine/multiple gpu training
        multi_training(args, config)
    else:
        # handle cpu & single gpu training
        single_training(args, config)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def single_training(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainLossEnc_plot = np.zeros(config.epochs)
    trainLossDec_plot = np.zeros(config.epochs)
    testLossEnc_plot = np.zeros(config.epochs)
    testLossDec_plot = np.zeros(config.epochs)

    trainset = DataLoader("train", config, args)
    trainloader = Data.DataLoader(trainset.dataset, batch_size = config.batch_size, shuffle = config.shuffle, num_workers = config.loadThread, pin_memory = True)
    testset = DataLoader("test", config, args)
    testloader = Data.DataLoader(testset.dataset, batch_size = config.batch_size, shuffle = config.shuffle, num_workers = config.loadThread, pin_memory = True)

    global net
    net=WNet(config.num_classes)
    net.apply(weights_init)
    summary(net)

    if torch.cuda.is_available():
        net=net.cuda()

    global criterion, gaussian
    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    gaussian = normal.Normal(0.0, 1.0)

    for epoch in range(config.epochs):  # loop over the dataset multiple times
        start_time = time.time()
        train_loss_enc, train_loss_dec = train_epoch(trainloader, optimizer)
        test_loss_enc, test_loss_dec = test_epoch(testloader, optimizer)
        lr_scheduler.step()

        end_time = time.time()

        # print statistics
        trainLossEnc_plot[epoch] = train_loss_enc
        trainLossDec_plot[epoch] = train_loss_dec
        testLossEnc_plot[epoch] = test_loss_enc
        testLossDec_plot[epoch] = test_loss_dec
        print("train_l_e: %5.3f, train_l_d: %5.3f ------ test_l_e: %5.3f, test_l_d: %5.3f" % (train_loss_enc, train_loss_dec, test_loss_enc, test_loss_dec))
        print("#%d: epoch_time: %8.2f, ~time_left: %8.2f" % (epoch, end_time-start_time, (end_time-start_time)*(config.epochs - epoch)))

    # save trained model
    # save under datapath + /result/sgd/model_name.pth
    model_name = 'lr_' + str(args.lr) + '.pth'
    loss_name = 'lr_' + str(args.lr) + '.png'
    savepath = os.path.join(args.datapath, 'result', 'sgd')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    torch.save(net.state_dict(), os.path.join(savepath, model_name))
    # plot loss
    timex = np.arange(0, epoch+1, 1)
    plt.plot(timex, trainLossEnc_plot[0:epoch+1], color='r', label='train_e')
    plt.plot(timex, trainLossDec_plot[0:epoch+1], color='m', label='train_d')
    plt.plot(timex, testLossEnc_plot[0:epoch+1], color='b', label='test_e')
    plt.plot(timex, testLossDec_plot[0:epoch+1], color='c', label='test_d')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(savepath, loss_name))

    print('Finished Training')

# Helpful documentation for single-node multi-gpu:
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
# https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py

# multi-node multi-gpu: https://leimao.github.io/blog/PyTorch-Distributed-Training/
def multi_training(args, config):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # setup WNet
    net = WNet(config.num_classes)
    net.cuda(args.local_rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    net.apply(weights_init)

    trainSet = DataLoader("train", config, args)
    testSet = DataLoader("test", config, args)
    trainSampler = torch.utils.data.distributed.DistributedSampler(trainSet.dataset)
    trainLoader = Data.DataLoader(dataset=trainSet.dataset, sampler=trainSampler, batch_size=config.batch_size // args.num_gpu,
                              drop_last=True, num_workers=config.loadThread, pin_memory=True)

    testSampler = torch.utils.data.distributed.DistributedSampler(testSet.dataset)
    testLoader = Data.DataLoader(dataset=testSet.dataset, sampler=testSampler, batch_size=config.batch_size // args.num_gpu,
                              drop_last=True, num_workers=config.loadThread, pin_memory=True)

    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    criterion0 = nn.BCELoss().cuda(args.local_rank)
    criterion1 = nn.MSELoss().cuda(args.local_rank)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-6, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    gaussian = normal.Normal(0.0, 1.0)

    trainLossEnc_plot = np.zeros(config.epochs)
    trainLossDec_plot = np.zeros(config.epochs)
    testLossEnc_plot = np.zeros(config.epochs)
    testLossDec_plot = np.zeros(config.epochs)
    #earlyStop = EarlyStop(patience=config.patience, threshold=config.threshold)

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
            labels = float(1.0) - labels

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
            # https://github.com/jvanvugt/pytorch-unet/blob/master/README.md ***** CRUCIAL CROPPING 
            enc = net(inputs, returns='enc')
            #define loss function here
            enc_loss=criterion1(enc, labels)  
            enc_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            dec = net(inputs, returns='dec')
            rec_loss=criterion1(dec, gaussian.cdf(inputs))
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
                labels = float(1.0) - labels

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
                enc_loss=criterion1(enc, labels)
                rec_loss=criterion1(dec, gaussian.cdf(inputs))

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
        torch.save(net.state_dict(), args.datapath + args.model_name)
        # plot loss
        timex = np.arange(0, epoch+1, 1)
        plt.plot(timex, trainLossEnc_plot[0:epoch+1], color='r', label='train_e')
        plt.plot(timex, trainLossDec_plot[0:epoch+1], color='m', label='train_d')
        plt.plot(timex, testLossEnc_plot[0:epoch+1], color='b', label='test_e')
        plt.plot(timex, testLossDec_plot[0:epoch+1], color='c', label='test_d')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(args.datapath + args.loss_name)


def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,:,pad[0]:-pad[1]]
    if pad[4]+pad[5] > 0:
        x = x[:,:,pad[4]:-pad[5],:,:]
    return x
 
def train_epoch(trainloader, optimizer):
    global net
    net.train()

    train_loss_enc = 0.0
    train_loss_dec = 0.0
    test_loss_enc = 0.0
    test_loss_dec = 0.0
    for batch_idx, data in enumerate(trainloader):
        # ge t the inputs; data is a list of [inputs, labels]
        inputs, labels, dec_labels = data

        # convert to cuda compatible
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()            
            dec_labels = dec_labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # https://github.com/jvanvugt/pytorch-unet/blob/master/README.md ***** CRUCIAL CROPPING 
        enc = net(inputs, returns='enc')
        #define loss function here

        enc_loss=criterion(enc, labels)
        enc_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        dec = net(inputs, returns='dec')

        rec_loss=criterion(dec, dec_labels)
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
            # ge t the inputs; data is a list of [inputs, labels]
            inputs, labels, dec_labels = data

            # convert to cuda compatible
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()            
                dec_labels = dec_labels.cuda()

            enc, dec = net(inputs)

            enc_loss=criterion(enc, labels)
            rec_loss=criterion(dec, dec_labels)

            test_loss_enc += enc_loss.item()
            test_loss_dec += rec_loss.item()

    return test_loss_enc / (batch_idx + 1), test_loss_dec / (batch_idx + 1)

if __name__ == "__main__":
    main()
