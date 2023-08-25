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
from util import EarlyStop
import time
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.utils.data as Data
from torchinfo import summary
from util import NCutLoss3D

def main():
    parser = argparse.ArgumentParser(description='DeepFDR using W-NET')
    parser.add_argument('--num_gpu', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--labelpath', default='./', type=str)
    parser.add_argument('--savepath', default='./', type=str)
    parser.add_argument('--loss', default='x_mse', type=str) # x_mse or pv_mse
    parser.add_argument('--mode', default=2, type=int) # 0 for train, 1 for inference, 2 for both
    args = parser.parse_args()

    config = Config()
    config.datapath = args.datapath
    config.labelpath = args.labelpath
    config.savepath = args.savepath

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if args.num_gpu > 1:
        # handle multiple machine/multiple gpu training
        multi_training(args, config)
    else:
        # handle cpu & single gpu training
        if args.mode == 0:
            single_training(args, config)
        elif args.mode == 1:
            model_name = 'lr_' + str(args.lr) + '.pth'
            compute_statistics(args, config, model_name)
        elif args.mode == 2:
            single_training(args, config)
            model_name = 'lr_' + str(args.lr) + '.pth'
            compute_statistics(args, config, model_name)

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

    global criterion1, criterion2, gaussian
    criterion1 = NCutLoss3D()
    criterion2 = nn.MSELoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    gaussian = normal.Normal(0.0, 1.0)

    for epoch in range(config.epochs):  # loop over the dataset multiple times
        start_time = time.time()
        train_loss_enc, train_loss_dec = train_epoch(trainloader, optimizer, args)
        test_loss_enc, test_loss_dec = test_epoch(testloader, optimizer, args)
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
    savepath = args.savepath
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
 
def train_epoch(trainloader, optimizer, args):
    global net
    net.train()

    train_loss_enc = 0.0
    train_loss_dec = 0.0
    test_loss_enc = 0.0
    test_loss_dec = 0.0
    for batch_idx, data in enumerate(trainloader):
        # ge t the inputs; data is a list of [inputs, labels]
        inputs, y_0, y_1 = data

        # convert to cuda compatible
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            y_0 = y_0.cuda()
            y_1 = y_1.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # https://github.com/jvanvugt/pytorch-unet/blob/master/README.md ***** CRUCIAL CROPPING 
        enc = net(inputs, returns='enc')

        enc_loss=criterion1(enc, y_0)
        enc_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        dec = net(inputs, returns='dec')

        if args.loss == 'x_mse':
            rec_loss=criterion2(dec, y_0)
        elif args.loss == 'pv_mse':
            dec = torch.sigmoid(dec)
            rec_loss=criterion2(dec, y_1)

        #loss_ratio = enc_loss.item() / rec_loss.item()
        #rec_loss = rec_loss * loss_ratio

        rec_loss.backward()
        optimizer.step()

        train_loss_enc += enc_loss.item()
        train_loss_dec += rec_loss.item()

    return train_loss_enc / (batch_idx + 1), train_loss_dec / (batch_idx + 1)


def test_epoch(testloader, optimizer, args):
    global net
    net.eval()

    test_loss_enc = 0.0
    test_loss_dec = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            # ge t the inputs; data is a list of [inputs, labels]
            inputs, y_0, y_1 = data

            # convert to cuda compatible
            if torch.cuda.is_available():
                inputs = inputs.cuda()            
                y_0 = y_0.cuda()
                y_1 = y_1.cuda()

            enc, dec = net(inputs, returns='both')

            enc_loss=criterion1(enc, y_0)
            if args.loss == 'x_mse':
                rec_loss=criterion2(dec, y_0)
            elif args.loss == 'pv_mse':
                dec = torch.sigmoid(dec)
                rec_loss=criterion2(dec, y_1)

            test_loss_enc += enc_loss.item()
            test_loss_dec += rec_loss.item()

    return test_loss_enc / (batch_idx + 1), test_loss_dec / (batch_idx + 1)

def p_lis(gamma_1, threshold=0.1, label=None, savepath=None):
    '''
    Rejection of null hypothesis are shown as 1, consistent with online BH, Q-value, smoothFDR methods.
    # LIS = P(theta = 0 | x)
    # gamma_1 = P(theta = 1 | x) = 1 - LIS
    '''
    gamma_1 = gamma_1.ravel()
    dtype = [('index', int), ('value', float)]
    size = gamma_1.shape[0]

    lis = np.zeros(size, dtype=dtype)
    for i in range(size):
        lis[i]['index'] = i
        lis[i]['value'] = 1 - gamma_1[i]
    # sort using lis values
    lis = np.sort(lis, order='value')
    # Data driven LIS-based FDR procedure
    sum = 0
    k = 0
    for j in range(len(lis)):
        sum += lis[j]['value']
        if sum > (j+1)*threshold:
            k = j
            break

    signal_lis = np.zeros(size)
    for j in range(k):
        index = lis[j]['index']
        signal_lis[index] = 1  

    if savepath is not None:
        np.save(savepath, signal_lis)

    if label is not None:
        # Compute FDR, FNR, ATP using LIS and Label
        # FDR -> (theta = 0) / num_rejected
        # FNR -> (theta = 1) / num_not_rejected
        # ATP -> (theta = 1) that is rejected
        num_rejected = k
        num_not_rejected = size - k
        fdr = 0
        fnr = 0
        atp = 0
        for i in range(size):
            if signal_lis[i] == 1: # rejected
                if label[i] == 0:
                    fdr += 1
                elif label[i] == 1:
                    atp += 1
            elif signal_lis[i] == 0: # not rejected
                if label[i] == 1:
                    fnr += 1

        if num_rejected == 0:
            fdr = 0
        else:
            fdr /= num_rejected

        if num_not_rejected == 0:
            fnr = 0
        else:
            fnr /= num_not_rejected

        return fdr, fnr, atp

def compute_statistics(args, config, model_name):
    # i'm running locally, so just test out 100, change this later
    data = np.load(os.path.join(args.datapath))['arr_0'].reshape((6000, 30,30,30))
    label = np.ravel(np.load(os.path.join(args.labelpath)))
    print(data.shape)
    print(label.shape)
    
    model = WNet(config.num_classes)
    model.cuda()
    if args.num_gpu > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(args.savepath, model_name)))
    model.eval()

    total_dl_fdr = 0
    total_dl_fnr = 0
    total_dl_atp = 0

    #load the files
    num_train = 1
    for i in range(0, num_train):
        input = torch.from_numpy(data[i]).float()
        p3d = (1, 1, 1, 1, 1, 1)
        input = F.pad(input, p3d, "constant", 0)
        input = input[None, :, :, :]
        input = input.unsqueeze(1)
        input = input.cuda()
        enc = model(input, returns='enc')
        dl_gamma = torch.squeeze(enc).detach().cpu().numpy()    

        fdr, fnr, atp = p_lis(gamma_1=dl_gamma, label=label, savepath=os.path.join(args.savepath, 'train_' + os.path.splitext(model_name)[0]) + '.npy')

        total_dl_fdr += fdr
        total_dl_fnr += fnr
        total_dl_atp += atp


    total_dl_fdr /= num_train
    total_dl_fnr /= num_train
    total_dl_atp /= num_train
   

    # Save final signal_file
    with open(os.path.join(args.savepath, 'train_' + os.path.splitext(model_name)[0]), 'w') as outfile:
        outfile.write('DL train:\n')
        outfile.write('fdr: ' + str(total_dl_fdr) + '\n')
        outfile.write('fnr: ' + str(total_dl_fnr) + '\n')
        outfile.write('atp: ' + str(total_dl_atp) + '\n')
  

    total_dl_fdr = 0
    total_dl_fnr = 0
    total_dl_atp = 0

    #load the files
    num_test = 1
    for i in range(num_train, num_train + num_test):
        input = torch.from_numpy(data[i]).float()
        p3d = (1, 1, 1, 1, 1, 1)
        input = F.pad(input, p3d, "constant", 0)
        input = input[None, :, :, :]
        input = input.unsqueeze(1)
        input = input.cuda()
        enc = model(input, returns='enc')
        dl_gamma = torch.squeeze(enc).detach().cpu().numpy()    

        fdr, fnr, atp = p_lis(gamma_1=dl_gamma, label=label, savepath=os.path.join(args.savepath, 'test_' + os.path.splitext(model_name)[0]) + '.npy')

        total_dl_fdr += fdr
        total_dl_fnr += fnr
        total_dl_atp += atp


    total_dl_fdr /= num_test
    total_dl_fnr /= num_test
    total_dl_atp /= num_test
    

    # Save final signal_file
    with open(os.path.join(args.savepath, 'test_' + os.path.splitext(model_name)[0]), 'w') as outfile:
        outfile.write('DL train:\n')
        outfile.write('fdr: ' + str(total_dl_fdr) + '\n')
        outfile.write('fnr: ' + str(total_dl_fnr) + '\n')
        outfile.write('atp: ' + str(total_dl_atp) + '\n')

if __name__ == "__main__":
    main()
