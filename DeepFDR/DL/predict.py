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

def p_lis(gamma_1, label):
    # LIS = P(theta = 0 | x)
    # gamma_1 = P(theta = 1 | x) = 1 - LIS
    dtype = [('index', float), ('value', float)]
    lis = np.zeros((30 * 30 * 30), dtype=dtype)
    for vx in range(30):
        for vy in range(30):
            for vk in range(30):
                index = (vk * 30 * 30) + (vy * 30) + vx
                lis[index]['index'] = index
                # can't just do this
                lis[index]['value'] = 1 - gamma_1[vx][vy][vk]
    # sort using lis values
    lis = np.sort(lis, order='value')
    # Data driven LIS-based FDR procedure
    sum = 0
    k = 0
    for j in range(len(lis)):
        sum += lis[j]['value']
        #print(j, lis[j]['value'])
        if sum > (j+1)*0.1:
            k = j
            break
    print("k", k)

    signal_lis = np.ones((30, 30, 30))
    for j in range(k):
        index = lis[j]['index']
        vk = index // (30*30)  # integer division
        index -= vk*30*30
        vy = index // 30  # integer division
        vx = index % 30
        vk = int(vk)
        vy = int(vy)
        vx = int(vx)
        signal_lis[vx][vy][vk] = 0  # reject these voxels, rest are 1

    # Compute FDR, FNR, ATP using LIS and Label
    # FDR -> (theta = 0) / num_rejected
    # FNR -> (theta = 1) / num_not_rejected
    # ATP -> (theta = 1) that is rejected
    num_rejected = k
    num_not_rejected = (30*30*30) - k
    fdr = 0
    fnr = 0
    atp = 0
    for i in range(30):
        for j in range(30):
            for k in range(30):
                if signal_lis[i][j][k] == 0: # rejected
                    if label[i][j][k] == 0:
                        fdr += 1
                    elif label[i][j][k] == 1:
                        atp += 1
                elif signal_lis[i][j][k] == 1: # not rejected
                    if label[i][j][k] == 1:
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


def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,:,pad[0]:-pad[1]]
    if pad[4]+pad[5] > 0:
        x = x[:,:,pad[4]:-pad[5],:,:]
    return x

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    model = WNet(config.num_classes)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(config.savepath))
    model.eval()

    total_dl_fdr = 0
    total_dl_fnr = 0
    total_dl_atp = 0
    total_orig_fdr = 0
    total_orig_fnr = 0
    total_orig_atp = 0

    #load the files
    for i in range(5000, 6000):
        dl_x = config.datapath + "test/" + str(i) + ".npy"
        orig_gamma = config.datapath + "test_target/" + str(i) + ".npy"
        input = np.load(dl_x).reshape((config.inputsize[0],config.inputsize[1],config.inputsize[2]))
        gamma = np.load(orig_gamma).reshape((30, 30, 30))
        label = np.loadtxt(config.label_datapath).reshape((30, 30, 30))
        input = torch.from_numpy(input).float()
        # zero pad input and label by 1 on each side to make it 32x32x32
        p3d = (1, 1, 1, 1, 1, 1)
        input = F.pad(input, p3d, "constant", 0)
        # add dimension to match conv3d weights
        input = input[None, :, :, :]
        input = input.unsqueeze(1)
        enc, dec = model(input)
        unpadded = unpad(enc, p3d)
        soft = torch.squeeze(unpadded).detach().cpu().numpy()    
        fdr, fnr, atp = p_lis(gamma_1=soft, label=label)
        fdr1, fnr1, atp1 = p_lis(gamma_1=gamma, label=label)

        total_dl_fdr += fdr
        total_dl_fnr += fnr
        total_dl_atp += atp
        total_orig_fdr += fdr1
        total_orig_fnr += fnr1
        total_orig_atp += atp1


    total_dl_fdr /= 1000
    total_dl_fnr /= 1000
    total_dl_atp /= 1000
    total_orig_fdr /= 1000
    total_orig_fnr /= 1000
    total_orig_atp /= 1000

    print("predicted: ")
    print("fdr: " + str(total_dl_fdr))        
    print("fnr: " + str(total_dl_fnr))        
    print("atp: " + str(total_dl_atp))     

    print("original: ")
    print("fdr: " + str(total_orig_fdr))        
    print("fnr: " + str(total_orig_fnr))        
    print("atp: " + str(total_orig_atp))     

           
if __name__ == '__main__':
    main()
