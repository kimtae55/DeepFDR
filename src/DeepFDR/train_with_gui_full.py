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
import multiprocessing
from multiprocessing import Queue, Event

# TODO: have two separate train(), one without these events for non_gui runs
def train(queue, continue_training_flag, save_flag, save_exit_flag, args):
    Config.datapath = args.datapath
    Config.labelpath = args.labelpath
    Config.savepath = args.savepath
    Config.replications = 1 # we're just visualizing for one replication, otherwise the for loop logic needs to be changed
                            # i'm not removing the for loop for now, to make it easier to read compared to train.main
    Config.epochs = 100 # This is a big number, to allow the user to save results then exit when satisfied without being limited by # of epochs. 

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    torch.cuda.manual_seed_all(Config.seed)

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
    for i in range(Config.replications):
        print('-----------------------------------------------')
        print('-------------------------------- RUN_NUMBER: ',i)
        print('-----------------------------------------------')

        if args.mode == 1:
            model_name = 'lr_' + str(args.lr) + '.pth' # change this to argparser
            r = compute_statistics(args, model_name) 
            print(f'epoch result: {r[0]},{r[1]},{r[2]}')

        elif args.mode == 2:
            Config.sample_number = i
            model_name = single_training(queue, continue_training_flag, save_flag, save_exit_flag, args, Config)
            r = compute_statistics(args, model_name)
            print(f'epoch result: {r[0]},{r[1]},{r[2]}')

            # aggregate results
            add_result(r[0], r[1], r[2], 0) # deepfdr

    end = time.time()
    print('DL computation time: ', (end-start)/Config.replications)

    for key, val in metric.items():
        print(key)
        for i in range(len(val)):
            print(f"{list(method_dict.keys())[i]} -- {val[i]}")

    # Save final signal_file
    with open(os.path.join(args.savepath, 'out_' + os.path.splitext(model_name)[0]) + '.txt', 'w') as outfile:
        outfile.write('DL:\n')
        mfdr, sfdr = np.mean(metric['fdr'][0]), np.std(metric['fdr'][0])
        mfnr, sfnr = np.mean(metric['fnr'][0]), np.std(metric['fnr'][0])
        matp, satp = np.mean(metric['atp'][0]), np.std(metric['atp'][0])
        outfile.write(f'fdr: {mfdr} ({sfdr})\n')
        outfile.write(f'fnr: {mfnr} ({sfnr})\n')
        outfile.write(f'atp: {matp} ({satp})\n')

    save_exit_flag.clear() # use this clear signal to stop the dash server

def weights_init(m): 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def single_training(queue, continue_training_flag, save_flag, save_exit_flag, args, Config):
    end_training = False

    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    if torch.cuda.is_available(): 
        torch.cuda.empty_cache() # clear gpu cache

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainLossEnc_plot = []
    trainLossDec_plot = []

    fdp_opt_1_plot = []
    fdp_opt_2_plot = []

    trainset = DataLoader("train", Config, args)
    trainloader = Data.DataLoader(trainset.dataset, batch_size = Config.batch_size, shuffle = Config.shuffle, num_workers = Config.loadThread, pin_memory = True)

    earlyStop = EarlyStop()

    global net
    net=WNet(Config.num_classes)
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

    for epoch in range(Config.epochs):  # loop over the dataset multiple times
        # Check if the user has clicked the "Continue Training" button
        continue_training_flag.clear()

        # event interactions 
        print("Training paused, click Continue to resume training...")
        while not continue_training_flag.is_set():
            if save_exit_flag.is_set() and epoch > 0:
                end_training = True
                print("Saving and exiting training...")
                break

            if save_flag.is_set() and epoch > 0:
                print('Saved!')
                model_name = 'lr_' + str(args.lr) + '_' + str(epoch-1) + '.pth'
                torch.save(net.state_dict(), os.path.join(args.savepath, model_name))
                save_flag.clear()

            continue_training_flag.wait(timeout=0.5) # Check every 0.5 second


        if end_training:
            print("Exiting out of training loop")
            break

        print("Training resumed...")

        start_time = time.time()
        tle, tld, fdps = train_epoch_whole(trainloader, optimizer_w, optimizer_e, args, Config)
        fdp_opt_1, fdp_opt_2, signal_lis = fdps
        end_time = time.time()

        # print statistics
        trainLossEnc_plot.append(tle)
        trainLossDec_plot.append(tld)
        fdp_opt_1_plot.append(fdp_opt_1)
        fdp_opt_2_plot.append(fdp_opt_2)
        print("#%d: epoch_time: %8.2f" % (epoch, end_time-start_time))
        print("\ttrain_l_e: %5.3f, train_l_d: %5.3f " % (tle, tld))
        print("\tfdp_opt_1: %5.3f" % (fdp_opt_1))
        print("\tfdp_opt_2: %5.3f" % (fdp_opt_2))

        # Get the Queue ready
        queue.put(np.array(trainLossEnc_plot))
        queue.put(np.array(fdp_opt_1_plot))
        queue.put(signal_lis)


    # save trained model
    model_name = 'lr_' + str(args.lr) + '_' + str(epoch-1) + '.pth'
    loss_name = 'lr_' + str(args.lr) + '_' + str(epoch-1) + '.png'

    torch.save(net.state_dict(), os.path.join(args.savepath, model_name))

    timex = list(range(epoch))

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharey=False)
    axs[0].plot(timex, trainLossEnc_plot, color='r', label='enc_loss')
    axs[0].plot(timex, trainLossDec_plot, color='orange', label='dec_loss')
    axs[1].plot(timex, fdp_opt_1_plot, color='g', label='FDP_p_option_1')
    axs[1].plot(timex, fdp_opt_2_plot, color='blue', label='FDP_p_option_2')
    axs[0].set(xlabel=r'epochs', ylabel='loss')
    axs[1].set(xlabel=r'epochs', ylabel='FDP')
    axs[0].grid(linewidth=0.5)
    axs[1].grid(linewidth=0.5)
    fig.legend(loc='lower right')
    plt.savefig(os.path.join(args.savepath, loss_name))

    print('Finishing Training...')
    return model_name

def compute_fdp_hat(data, p, qv, args):
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
    qv_signals = np.where(qv_np <= Config.threshold, 1, 0)
    p_np = torch.squeeze(p).detach().cpu().numpy().ravel()  
    label = np.ravel(np.load(Config.labelpath)[Config.cluster_number])

    size = gamma_1.size
    fdr, fnr, atp = p_lis(gamma_1=gamma_1, threshold=Config.threshold, label=label, flip=True)
    print('flip=True STATS: ', fdr, fnr, atp)
    fdr, fnr, atp = p_lis(gamma_1=gamma_1, threshold=Config.threshold, label=label, flip=False)
    print('flip=False STATS: ', fdr, fnr, atp)

    k_flip, lis_flip = p_lis(gamma_1=gamma_1, threshold=Config.threshold, flip=True)
    k_noflip, lis_noflip = p_lis(gamma_1=gamma_1, threshold=Config.threshold, flip=False)

    sl_flip = np.zeros(gamma_1.size)
    sl_flip[lis_flip[:k_flip]['index']] = 1
    sl_noflip = np.zeros(gamma_1.size)
    sl_noflip[lis_noflip[:k_noflip]['index']] = 1

    dice_flip = dice(qv_signals, sl_flip)
    dice_noflip = dice(qv_signals, sl_noflip)

    print('DICE:', dice_noflip, dice_flip)
    if dice_noflip > dice_flip: # this one is correct 
        lis = lis_noflip
        k = k_noflip
    else: # flipped one is correct
        lis = lis_flip
        k = k_flip

    # get k
    lis = np.sort(lis, order='value')
    cumulative_sum = np.cumsum(lis[:]['value'])
    k = np.argmax(cumulative_sum > (np.arange(len(lis)) + 1)*Config.threshold)

    signal_lis = np.zeros(size)
    signal_lis[lis[:k]['index']] = 1

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

    return (fdp_opt_1, fdp_opt_2, signal_lis)

def train_epoch_whole(trainloader, optimizer_w, optimizer_e, args, Config):
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
    fdps = compute_fdp_hat(inputs, y_1, y_2, args)

    return train_loss_enc, train_loss_dec, fdps


def compute_statistics(args, model_name):
    data = np.load(os.path.join(args.datapath))[Config.sample_number:Config.sample_number+1].reshape((-1,)+Config.inputsize)
    label = np.ravel(np.load(Config.labelpath)[Config.cluster_number])
    
    def predict_for_wnet():
        # Model setup 
        model = WNet(Config.num_classes)
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

    q_rejected = compute_qval(data, Config.threshold)
    qv_signals = np.where(q_rejected <= Config.threshold, 1, 0)

    gamma_1 = predict_for_wnet()
    k_flip, lis_flip = p_lis(gamma_1=gamma_1, threshold=Config.threshold, flip=True)
    k_noflip, lis_noflip = p_lis(gamma_1=gamma_1, threshold=Config.threshold, flip=False)

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

    fdr, fnr, atp = p_lis(gamma_1=gamma_1, threshold=Config.threshold, label=label, savepath=os.path.join(args.savepath, 'auto_' + os.path.splitext(model_name)[0]), flip=flip)
    return (fdr, fnr, atp)

