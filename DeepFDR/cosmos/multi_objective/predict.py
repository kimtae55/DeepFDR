import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import WNet
import time
import matplotlib.pyplot as plt
from methods.cosmos.cosmos import Upsampler
from torchinfo import summary
from utils import circle_points

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
        if sum > (j+1)*0.1:
            k = j
            break

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
    parser = argparse.ArgumentParser(description='DeepFDR using COSMOS')
    parser.add_argument('--model_name', default='wnet.pth', type=str)
    parser.add_argument('--num_gpu', default=1, type=int)
    parser.add_argument('--datapath', type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    # load data, groundtruth, and gamma (logits)
    data = np.load(os.path.join(args.datapath, 'data', 'data.npz'))['arr_0']
    label = np.loadtxt(os.path.join(os.path.abspath(os.path.join(args.datapath, os.pardir)), 'label', "label.txt")).reshape((30, 30, 30))
    gamma = np.load(os.path.join(args.datapath, 'data', 'label.npz'))['arr_0']

    baseModel = WNet()
    K = 2
    lamda = 2
    n_test_rays = 10
    dim = [1, 32, 32, 32]
    dim[0] = dim[0] + K
    test_rays = circle_points(n_test_rays, dim=K)

    model = Upsampler(K, baseModel, dim)
    summary(model)

    if args.num_gpu > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(args.datapath, 'result', 'cosmos', args.model_name), map_location=map_location))
    model.eval()

    total_dl_fdr = np.zeros(n_test_rays)
    total_dl_fnr = np.zeros(n_test_rays)
    total_dl_atp = np.zeros(n_test_rays)
    #lis_diff = np.zeros((n_test_rays, 30,30,30))

    #load the files
    for i in range(0, 5000):
        input = torch.from_numpy(data[i]).float()
        # zero pad input and label by 1 on each side to make it 32x32x32
        p3d = (1, 1, 1, 1, 1, 1)
        input = F.pad(input, p3d, "constant", 0)
        # add dimension to match conv3d weights
        input = input[None, :, :, :]
        input = input.unsqueeze(1)
        input = input

        batch = dict(data=input)

        result = None
        for j, ray in enumerate(test_rays):
            ray = torch.from_numpy(ray.astype(np.float32))
            ray /= ray.sum()

            batch['alpha'] = ray
            result = model(batch)

            lis = torch.squeeze(result['logits_l']).detach().numpy()    
            dl_gamma = float(1.0) - lis
            #lis_diff[j] += np.abs(dl_gamma/gamma[i])

            fdr, fnr, atp = p_lis(gamma_1=dl_gamma, label=label)

            total_dl_fdr[j] += fdr
            total_dl_fnr[j] += fnr
            total_dl_atp[j] += atp

    #lis_diff /= 5000
    total_dl_fdr /= 5000
    total_dl_fnr /= 5000
    total_dl_atp /= 5000

    # Save final signal_file
    with open(os.path.join(args.datapath, 'result', 'cosmos', 'train_' + os.path.splitext(args.model_name)[0]), 'w') as outfile:
        outfile.write('DL train:\n')
 
        for i, ray in enumerate(test_rays):
            outfile.write('Ray: ' + str(ray) + '\n')
            outfile.write('fdr: ' + str(total_dl_fdr[i]) + '\n')
            outfile.write('fnr: ' + str(total_dl_fnr[i]) + '\n')
            outfile.write('atp: ' + str(total_dl_atp[i]) + '\n')   

    total_dl_fdr = np.zeros(n_test_rays)
    total_dl_fnr = np.zeros(n_test_rays)
    total_dl_atp = np.zeros(n_test_rays)
    #lis_diff = np.zeros((n_test_rays, 30,30,30))

    #load the files
    for i in range(5000, 6000):
        input = torch.from_numpy(data[i]).float()
        # zero pad input and label by 1 on each side to make it 32x32x32
        p3d = (1, 1, 1, 1, 1, 1)
        input = F.pad(input, p3d, "constant", 0)
        # add dimension to match conv3d weights
        input = input[None, :, :, :]
        input = input.unsqueeze(1)
        input = input

        batch = dict(data=input)

        result = None
        for j, ray in enumerate(test_rays):
            ray = torch.from_numpy(ray.astype(np.float32))
            ray /= ray.sum()

            batch['alpha'] = ray
            result = model(batch)

            lis = torch.squeeze(result['logits_l']).detach().numpy()    
            dl_gamma = float(1.0) - lis
            #lis_diff[j] += np.abs(dl_gamma/gamma[i])

            fdr, fnr, atp = p_lis(gamma_1=dl_gamma, label=label)

            total_dl_fdr[j] += fdr
            total_dl_fnr[j] += fnr
            total_dl_atp[j] += atp

    #lis_diff /= 5000
    total_dl_fdr /= 1000
    total_dl_fnr /= 1000
    total_dl_atp /= 1000

    # Save final signal_file
    with open(os.path.join(args.datapath, 'result', 'cosmos', 'test_' + os.path.splitext(args.model_name)[0]), 'w') as outfile:
        outfile.write('DL train:\n')
 
        for i, ray in enumerate(test_rays):
            outfile.write('Ray: ' + str(ray) + '\n')
            outfile.write('fdr: ' + str(total_dl_fdr[i]) + '\n')
            outfile.write('fnr: ' + str(total_dl_fnr[i]) + '\n')
            outfile.write('atp: ' + str(total_dl_atp[i]) + '\n')   
           
if __name__ == '__main__':
    main()
