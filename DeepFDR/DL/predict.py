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
import scipy.stats
import sys
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy import interpolate

def q_estimate(pv, m=None, verbose=False, lowmem=False, pi0=None):
    """
    Estimates q-values from p-values
    Args
    =====
    m: number of tests. If not specified m = pv.size
    verbose: print verbose messages? (default False)
    lowmem: use memory-efficient in-place algorithm
    pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.
         For most GWAS this is not necessary, since pi0 is extremely likely to be
         1
    """
    assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

    original_shape = pv.shape
    pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

    if m is None:
        m = float(len(pv))
    else:
        # the user has supplied an m
        m *= 1.0

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(pv > i).sum() for i in np.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        # fit natural cubic spline
        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)
        if verbose:
            print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

        if pi0 > 1:
            if verbose:
                print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            pi0 = 1.0

    assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    if lowmem:
        # low memory version, only uses 1 pv and 1 qv matrices
        qv = np.zeros((len(pv),))
        last_pv = pv.argmax()
        qv[last_pv] = (pi0*pv[last_pv]*m)/float(m)
        pv[last_pv] = -np.inf
        prev_qv = last_pv
        for i in range(int(len(pv))-2, -1, -1):
            cur_max = pv.argmax()
            qv_i = (pi0*m*pv[cur_max]/float(i+1))
            pv[cur_max] = -np.inf
            qv_i1 = prev_qv
            qv[cur_max] = min(qv_i, qv_i1)
            prev_qv = qv[cur_max]

    else:
        p_ordered = np.argsort(pv)
        pv = pv[p_ordered]
        qv = pi0 * m/len(pv) * pv
        qv[-1] = min(qv[-1], 1.0)

        for i in range(len(pv)-2, -1, -1):
            qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

        # reorder qvalues
        qv_temp = qv.copy()
        qv = np.zeros_like(qv)
        qv[p_ordered] = qv_temp

    # reshape qvalues
    qv = qv.reshape(original_shape)

    return qv


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

def compute_bh_and_qval(x, label):
    # calculate p_value from z-score
    p_value = scipy.stats.norm.sf(np.fabs(x))*2.0 # two-sided tail, calculates 1-cdf
    p_value = np.ravel(p_value)

    ############################ BH method ##############################################
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_value, alpha=0.1, method='fdr_bh')

    ############################ Q-value method ##############################################
    qv = q_estimate(p_value).reshape((30, 30, 30))

    reject = reject.reshape((30, 30, 30))

    num_rejected = np.count_nonzero(reject)
    num_not_rejected = (30 * 30 * 30) - num_rejected
    fdr = 0
    fnr = 0
    atp = 0

    fdr_q = 0
    fnr_q = 0
    atp_q = 0
    num_rejected_q = 0
    for i in range(30):
        for j in range(30):
            for k in range(30):
                if reject[i][j][k] == 1:  # rejected
                    if label[i][j][k] == 0:
                        fdr += 1
                    elif label[i][j][k] == 1:
                        atp += 1
                elif reject[i][j][k] == 0:  # not rejected
                    if label[i][j][k] == 1:
                        fnr += 1
                if qv[i][j][k] < 0.1: # reject
                    num_rejected_q += 1
                    if label[i][j][k] == 0:
                        fdr_q += 1
                    elif label[i][j][k] == 1:
                        atp_q += 1
                else: # not rejected
                    if label[i][j][k] == 1:
                        fnr_q += 1

    if num_rejected == 0:
        fdr = 0
    else:
        fdr /= num_rejected

    if num_not_rejected == 0:
        fnr = 0
    else:
        fnr /= num_not_rejected

    if num_rejected_q == 0:
        fdr_q = 0
    else:
        fdr_q /= num_rejected_q

    if (30 * 30 * 30) - num_rejected_q == 0:
        fnr_q = 0
    else:
        fnr_q /= (30 * 30 * 30) - num_rejected_q

    return fdr, fnr, atp, fdr_q, fnr_q, atp_q   

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,:,pad[0]:-pad[1]]
    if pad[4]+pad[5] > 0:
        x = x[:,:,pad[4]:-pad[5],:,:]
    return x

def main():
    parser = argparse.ArgumentParser(description='DeepFDR using W-NET')
    parser.add_argument('--model_name', default='wnet.pth', type=str)
    parser.add_argument('--num_gpu', default=1, type=int)
    parser.add_argument('--datapath', type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()

    model = WNet(config.num_classes)
    model.cuda()
    if args.num_gpu > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(args.datapath, 'result', 'sgd', args.model_name)))
    model.eval()

    total_bh_fdr = 0
    total_bh_fnr = 0
    total_bh_atp = 0
    total_q_fdr = 0
    total_q_fnr = 0
    total_q_atp = 0
    total_dl_fdr = 0
    total_dl_fnr = 0
    total_dl_atp = 0
    total_orig_fdr = 0
    total_orig_fnr = 0
    total_orig_atp = 0
    lis_diff = np.zeros((30,30,30))

    data = np.load(os.path.join(args.datapath, 'data', 'data.npz'))['arr_0']
    label = np.loadtxt(os.path.join(os.path.abspath(os.path.join(args.datapath, os.pardir)), 'label', "label.txt")).reshape((30, 30, 30))
    gamma = np.load(os.path.join(args.datapath, 'data', 'label.npz'))['arr_0']

    #load the files
    for i in range(0, 5000):
        input = torch.from_numpy(data[i]).float()
        p3d = (1, 1, 1, 1, 1, 1)
        input = F.pad(input, p3d, "constant", 0)
        input = input[None, :, :, :]
        input = input.unsqueeze(1)
        input = input.cuda()
        enc = model(input, returns='enc')
        lis = torch.squeeze(enc).detach().cpu().numpy()    
        dl_gamma = float(1.0) - lis
        lis_diff += np.abs(dl_gamma/gamma[i])

        fdr, fnr, atp = p_lis(gamma_1=dl_gamma, label=label)
        fdr1, fnr1, atp1 = p_lis(gamma_1=gamma[i], label=label)
        fdr_bh, fnr_bh, atp_bh, fdr_q, fnr_q, atp_q = compute_bh_and_qval(data[i], label)

        total_bh_fdr += fdr_bh
        total_bh_fnr += fnr_bh
        total_bh_atp += atp_bh
        total_q_fdr += fdr_q
        total_q_fnr += fnr_q
        total_q_atp += atp_q

        total_dl_fdr += fdr
        total_dl_fnr += fnr
        total_dl_atp += atp
        total_orig_fdr += fdr1
        total_orig_fnr += fnr1
        total_orig_atp += atp1

    lis_diff /= 5000
    total_dl_fdr /= 5000
    total_dl_fnr /= 5000
    total_dl_atp /= 5000
    total_orig_fdr /= 5000
    total_orig_fnr /= 5000
    total_orig_atp /= 5000
    total_bh_fdr /= 5000
    total_bh_fnr /= 5000
    total_bh_atp /= 5000
    total_q_fdr /= 5000
    total_q_fnr /= 5000
    total_q_atp /= 5000

    # Save final signal_file
    with open(os.path.join(args.datapath, 'result', 'sgd', 'train_' + os.path.splitext(args.model_name)[0]), 'w') as outfile:
        outfile.write('DL train:\n')
        outfile.write('fdr: ' + str(total_dl_fdr) + '\n')
        outfile.write('fnr: ' + str(total_dl_fnr) + '\n')
        outfile.write('atp: ' + str(total_dl_atp) + '\n')
        outfile.write('Model2 train:\n')
        outfile.write('fdr: ' + str(total_orig_fdr) + '\n')
        outfile.write('fnr: ' + str(total_orig_fnr) + '\n')
        outfile.write('atp: ' + str(total_orig_atp) + '\n')        
        outfile.write('Q-value train:\n')
        outfile.write('fdr: ' + str(total_q_fdr) + '\n')
        outfile.write('fnr: ' + str(total_q_fnr) + '\n')
        outfile.write('atp: ' + str(total_q_atp) + '\n')
        outfile.write('BH train:\n')
        outfile.write('fdr: ' + str(total_bh_fdr) + '\n')
        outfile.write('fnr: ' + str(total_bh_fnr) + '\n')
        outfile.write('atp: ' + str(total_bh_atp) + '\n')
        outfile.write('Mean abs lis_diff between Model2 and DL:\n')
        for data_slice in lis_diff:
            np.savetxt(outfile, data_slice, fmt='%-8.4f')
            outfile.write('# New z slice\n')     

    total_bh_fdr = 0
    total_bh_fnr = 0
    total_bh_atp = 0
    total_q_fdr = 0
    total_q_fnr = 0
    total_q_atp = 0
    total_dl_fdr = 0
    total_dl_fnr = 0
    total_dl_atp = 0
    total_orig_fdr = 0
    total_orig_fnr = 0
    total_orig_atp = 0
    lis_diff = np.zeros((30,30,30))

    #load the files
    for i in range(5000, 6000):
        input = torch.from_numpy(data[i]).float()
        p3d = (1, 1, 1, 1, 1, 1)
        input = F.pad(input, p3d, "constant", 0)
        input = input[None, :, :, :]
        input = input.unsqueeze(1)
        input = input.cuda()
        enc = model(input, returns='enc')
        lis = torch.squeeze(enc).detach().cpu().numpy()    
        dl_gamma = float(1.0) - lis
        lis_diff += np.abs(dl_gamma/gamma[i])

        fdr, fnr, atp = p_lis(gamma_1=dl_gamma, label=label)
        fdr1, fnr1, atp1 = p_lis(gamma_1=gamma[i], label=label)
        fdr_bh, fnr_bh, atp_bh, fdr_q, fnr_q, atp_q = compute_bh_and_qval(data[i], label)

        total_bh_fdr += fdr_bh
        total_bh_fnr += fnr_bh
        total_bh_atp += atp_bh
        total_q_fdr += fdr_q
        total_q_fnr += fnr_q
        total_q_atp += atp_q

        total_dl_fdr += fdr
        total_dl_fnr += fnr
        total_dl_atp += atp
        total_orig_fdr += fdr1
        total_orig_fnr += fnr1
        total_orig_atp += atp1

    lis_diff /= 1000
    total_dl_fdr /= 1000
    total_dl_fnr /= 1000
    total_dl_atp /= 1000
    total_orig_fdr /= 1000
    total_orig_fnr /= 1000
    total_orig_atp /= 1000
    total_bh_fdr /= 1000
    total_bh_fnr /= 1000
    total_bh_atp /= 1000
    total_q_fdr /= 1000
    total_q_fnr /= 1000
    total_q_atp /= 1000

    # Save final signal_file
    with open(os.path.join(args.datapath, 'result', 'sgd', 'test_' + os.path.splitext(args.model_name)[0]), 'w') as outfile:
        outfile.write('DL train:\n')
        outfile.write('fdr: ' + str(total_dl_fdr) + '\n')
        outfile.write('fnr: ' + str(total_dl_fnr) + '\n')
        outfile.write('atp: ' + str(total_dl_atp) + '\n')
        outfile.write('Model2 train:\n')
        outfile.write('fdr: ' + str(total_orig_fdr) + '\n')
        outfile.write('fnr: ' + str(total_orig_fnr) + '\n')
        outfile.write('atp: ' + str(total_orig_atp) + '\n')        
        outfile.write('Q-value train:\n')
        outfile.write('fdr: ' + str(total_q_fdr) + '\n')
        outfile.write('fnr: ' + str(total_q_fnr) + '\n')
        outfile.write('atp: ' + str(total_q_atp) + '\n')
        outfile.write('BH train:\n')
        outfile.write('fdr: ' + str(total_bh_fdr) + '\n')
        outfile.write('fnr: ' + str(total_bh_fnr) + '\n')
        outfile.write('atp: ' + str(total_bh_atp) + '\n')     
        outfile.write('Mean abs lis_diff between Model2 and DL:\n')
        for data_slice in lis_diff:
            np.savetxt(outfile, data_slice, fmt='%-8.4f')
            outfile.write('# New z slice\n')     
           
if __name__ == '__main__':
    main()
