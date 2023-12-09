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
import random
from util import SoftNCutLoss3D, FDR_MSELoss, compute_qval, p_lis, dice
import time 

def main(args):
    config = Config()
    config.datapath = args.datapath
    config.labelpath = args.labelpath
    config.savepath = args.savepath

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    labelname_noext = '.'.join(config.labelpath.split('/')[-1].split('.')[:-1])

    method_dict = {
        'deepfdr': 0,
        # put other methods here if desired
    }
    
    metric = {
        'fdr': [[] for _ in range(len(method_dict))],
        'fnr': [[] for _ in range(len(method_dict))],
        'atp': [[] for _ in range(len(method_dict))]
    }
    def add_result(fdr, fnr, atp, method_index):
        metric['fdr'][method_index].append(fdr)
        metric['fnr'][method_index].append(fnr)
        metric['atp'][method_index].append(atp)

    start = time.time()
    for i in range(config.replications):
        print('-----------------------------------------------')
        print('-------------------------------- RUN_NUMBER: ',i)
        print('-----------------------------------------------')

        if args.mode == 1:
            model_name = 'lr_' + str(args.lr) + '.pth'
            r = compute_statistics(args, config, model_name)
            print(f'epoch result: {r[0]},{r[1]},{r[2]}')

        elif args.mode == 2:
            config.sample_number = i
            single_training(args, config)
            model_name = 'lr_' + str(args.lr) + '.pth'
            r = compute_statistics(args, config, model_name)
            print(f'epoch result: {r[0]},{r[1]},{r[2]}')

            # aggregate results
            add_result(r[0], r[1], r[2], 0) # deepfdr

    end = time.time()
    print('DL computation time: ', (end-start)/config.replications)

    for key, val in metric.items():
        print(key)
        for i in range(len(val)):
            print(f"{list(method_dict.keys())[i]} -- {val[i]}")

    # Save final signal_file
    with open(os.path.join(args.savepath, 'out_' + labelname_noext + '_' + os.path.splitext(model_name)[0]) + '.txt', 'w') as outfile:
        outfile.write('DL:\n')
        mfdr, sfdr = np.mean(metric['fdr'][0]), np.std(metric['fdr'][0])
        mfnr, sfnr = np.mean(metric['fnr'][0]), np.std(metric['fnr'][0])
        matp, satp = np.mean(metric['atp'][0]), np.std(metric['atp'][0])
        outfile.write(f'fdr: {mfdr} ({sfdr})\n')
        outfile.write(f'fnr: {mfnr} ({sfnr})\n')
        outfile.write(f'atp: {matp} ({satp})\n')

def weights_init(m): 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def single_training(args, config):
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache() # clear gpu cache

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainLossEnc_plot = np.zeros(config.epochs)
    trainLossDec_plot = np.zeros(config.epochs)

    fdp_opt_1_plot = np.zeros(config.epochs)
    fdp_opt_2_plot = np.zeros(config.epochs)

    trainset = DataLoader("train", config, args)
    trainloader = Data.DataLoader(trainset.dataset, batch_size = config.batch_size, shuffle = config.shuffle, num_workers = config.loadThread, pin_memory = True)

    earlyStop = EarlyStop()

    global net
    net=WNet(config.num_classes)
    net.apply(weights_init)
    summary(net)

    if torch.cuda.is_available():
        net=net.cuda()

    global criterion1, criterion2, criterion3
    criterion1 = SoftNCutLoss3D()
    criterion2 = FDR_MSELoss()
    optimizer_w = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)
    optimizer_e = torch.optim.SGD(net.UEnc.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)
    lr_scheduler_w = torch.optim.lr_scheduler.ExponentialLR(optimizer_w, gamma=1.0)
    lr_scheduler_e = torch.optim.lr_scheduler.ExponentialLR(optimizer_e, gamma=1.0)

    for epoch in range(config.epochs):  # loop over the dataset multiple times
        start_time = time.time()
        tle, tld, fdps = train_epoch_whole(trainloader, optimizer_w, optimizer_e, args, config)
        fdp_opt_1, fdp_opt_2 = fdps
        #lr_scheduler_w.step()
        #lr_scheduler_e.step()
        end_time = time.time()

        # print statistics
        trainLossEnc_plot[epoch] = tle
        trainLossDec_plot[epoch] = tld
        fdp_opt_1_plot[epoch] = fdp_opt_1
        fdp_opt_2_plot[epoch] = fdp_opt_2
        print("#%d: epoch_time: %8.2f, ~time_left: %8.2f" % (epoch, end_time-start_time, (end_time-start_time)*(config.epochs - epoch)))
        print("\ttrain_l_e: %5.3f, train_l_d: %5.3f " % (tle, tld))
        print("\tfdp_opt_1: %5.3f" % (fdp_opt_1))
        print("\tfdp_opt_2: %5.3f" % (fdp_opt_2))

        # Early stop
        #if earlyStop(train_loss_enc):
        #    print("Early stop activated.")
        #    break

    # save trained model
    model_name = 'lr_' + str(args.lr) + '.pth'
    loss_name = 'lr_' + str(args.lr) + '.png'
    savepath = args.savepath
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    torch.save(net.state_dict(), os.path.join(savepath, model_name))

    timex = np.arange(0, epoch+1, 1)

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharey=False)
    axs[0].plot(timex, trainLossEnc_plot[0:epoch+1], color='r', label='enc_loss')
    axs[0].plot(timex, trainLossDec_plot[0:epoch+1], color='orange', label='dec_loss')
    axs[1].plot(timex, fdp_opt_1_plot[0:epoch+1], color='g', label='FDP_p_option_1')
    axs[1].plot(timex, fdp_opt_2_plot[0:epoch+1], color='blue', label='FDP_p_option_2')
    axs[0].set(xlabel=r'epochs', ylabel='loss')
    axs[1].set(xlabel=r'epochs', ylabel='FDP')
    axs[0].grid(linewidth=0.5)
    axs[1].grid(linewidth=0.5)
    fig.legend(loc='lower right')
    plt.savefig(os.path.join(savepath, loss_name))

    print('Finished Training')

def compute_fdp_hat(data, p, qv, args, config):
    """
    Computes an overestimation of fdp using two different methods. 
    These estimations can be used along with early stop to monitor training internally. 
    Provides good indications for divergence/training difficulty
    """
    global net
    net.eval() # crucial!
    enc = net(data, returns='enc')
    gamma_1 = torch.squeeze(enc).detach().cpu().numpy().ravel()
    qv_np = torch.squeeze(qv).detach().cpu().numpy().ravel()
    qv_signals = np.where(qv_np <= config.threshold, 1, 0)
    p_np = torch.squeeze(p).detach().cpu().numpy().ravel()  
    label = np.ravel(np.load(config.labelpath)[config.cluster_number])

    size = gamma_1.size

    k_flip, lis_flip = p_lis(gamma_1=gamma_1, threshold=config.threshold, flip=True)
    k_noflip, lis_noflip = p_lis(gamma_1=gamma_1, threshold=config.threshold, flip=False)

    sl_flip = np.zeros(gamma_1.size)
    sl_flip[lis_flip[:k_flip]['index']] = 1
    sl_noflip = np.zeros(gamma_1.size)
    sl_noflip[lis_noflip[:k_noflip]['index']] = 1

    dice_flip = dice(qv_signals, sl_flip)
    dice_noflip = dice(qv_signals, sl_noflip)

    if dice_noflip > dice_flip: # this one is correct 
        lis = lis_noflip
        k = k_noflip
    else: # flipped one is correct
        lis = lis_flip
        k = k_flip

    # get k
    lis = np.sort(lis, order='value')
    cumulative_sum = np.cumsum(lis[:]['value'])
    k = np.argmax(cumulative_sum > (np.arange(len(lis)) + 1)*config.threshold)

    # fdp_opt_1
    m = 27000 # 439758 for real data analysis Don't forget!!!
    lambda_m = 1 - 2*m**(-1/3)/np.log(np.log(m))
    num_p_lt_labmda = np.sum(np.where(p_np < lambda_m, 1, 0))
    rx = k
    sigx = np.sum(p_np[lis[:k]['index']])
    p_h_0 = (m-num_p_lt_labmda+1) / (m*(1-lambda_m))
    fdp_opt_1 = p_h_0 * sigx / rx if rx > 0 else 0

    # fdp_opt_2
    def kth_smallest(arr, k):
        arr_c = arr.copy()
        arr_c.sort()  # Sort the array in ascending order
        return arr_c[int(k - 1)]
    m = 27000
    tau_m = np.floor(m**(0.5))
    p_mtau = kth_smallest(p_np, m-tau_m)
    rx = k
    sigx = np.sum(p_np[lis[:k]['index']])
    p_h_0 = (tau_m + 1) / (m*(1-p_mtau))
    fdp_opt_2 = p_h_0 * sigx / rx if rx > 0 else 0

    return (fdp_opt_1, fdp_opt_2)

def train_epoch_whole(trainloader, optimizer_w, optimizer_e, args, config):
    global net
    net.train()

    train_loss_enc = 0.0
    train_loss_dec = 0.0
    for batch_idx, data in enumerate(trainloader):
        # ge t the inputs; data is a list of [inputs, labels]
        inputs, y_0, y_1, y_2 = data

        # convert to cuda compatible
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            y_0 = y_0.cuda()
            y_1 = y_1.cuda()
            y_2 = y_2.cuda()
        # zero the parameter gradients
        optimizer_e.zero_grad()

        enc = net(inputs, returns='enc')

        enc_loss=criterion1(y_0, enc)

        enc_loss.backward()
        optimizer_e.step()

        optimizer_w.zero_grad()
        dec = net(inputs, returns='dec', qv=y_2)

        if args.loss == 'x_mse':
            rec_loss=criterion2(dec, y_0)
        elif args.loss == 'pv_mse':
            dec = torch.sigmoid(dec)
            rec_loss=criterion2(y_1, dec)

        rec_loss.backward()
        optimizer_w.step()

        train_loss_enc += enc_loss.item()
        train_loss_dec += rec_loss.item()

    net.eval()
    fdps = compute_fdp_hat(inputs, y_1, y_2, args, config)

    return train_loss_enc, train_loss_dec, fdps


def compute_statistics(args, config, model_name):
    data = np.load(os.path.join(args.datapath))[config.sample_number:config.sample_number+1].reshape((-1,)+config.inputsize)
    label = np.ravel(np.load(config.labelpath)[config.cluster_number])
    
    def predict_for_wnet():
        # Model setup 
        model = WNet(config.num_classes)
        if torch.cuda.is_available():
            model = model.cuda()

        model.load_state_dict(torch.load(os.path.join(args.savepath, model_name)))
        model.eval()

        # Inference Step
        input = torch.from_numpy(data).float()
        p3d = (1, 1, 1, 1, 1, 1)
        input = F.pad(input, p3d, "constant", 0)
        input = input[None, :, :, :]
        if torch.cuda.is_available():
            input = input.cuda()
        enc = model(input, returns='enc')
        dl_gamma = torch.squeeze(enc).detach().cpu().numpy()  
        return dl_gamma 

    q_rejected = compute_qval(data, config.threshold)
    qv_signals = np.where(q_rejected <= config.threshold, 1, 0)

    gamma_1 = predict_for_wnet()
    k_flip, lis_flip = p_lis(gamma_1=gamma_1, threshold=config.threshold, flip=True)
    k_noflip, lis_noflip = p_lis(gamma_1=gamma_1, threshold=config.threshold, flip=False)

    sl_flip = np.zeros(gamma_1.size)
    sl_flip[lis_flip[:k_flip]['index']] = 1
    sl_noflip = np.zeros(gamma_1.size)
    sl_noflip[lis_noflip[:k_noflip]['index']] = 1

    dice_flip = dice(qv_signals, sl_flip)
    dice_noflip = dice(qv_signals, sl_noflip)

    if dice_noflip < dice_flip:  
        flip = True
    else: 
        flip = False

    fdr, fnr, atp = p_lis(gamma_1=gamma_1, threshold=config.threshold, label=label, savepath=os.path.join(args.savepath, 'auto_' + os.path.splitext(model_name)[0]), flip=flip)
    return (fdr, fnr, atp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepFDR using W-NET')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--labelpath', default='./', type=str)
    parser.add_argument('--savepath', default='./', type=str)
    parser.add_argument('--loss', default='pv_mse', type=str) # x_mse or pv_mse
    parser.add_argument('--mode', default=2, type=int) # 0 for train, 1 for inference, 2 for both
    args = parser.parse_args()
    main(args)
