import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys
import h5py
import math


if __name__ == "__main__":
    x_mu = ['-4.0', '-3.5', '-3.0', '-2.5','-2.0', '-1.5', '-1.0']
    x_sigma = ['0.125', '0.25', '0.5', '1.0' ,'2.0', '4.0', '8.0']
    ''' for 10 rep simulation
    datapath = os.path.join(os.getcwd(), '../data/model1')

    sigma_files = [list(range(1,11)), list(range(11,21)), list(range(21,31)), list(range(31,41)), list(range(41,51)), list(range(51,61)), list(range(61,71))]
    mu_files = [list(range(71,81)), list(range(81,91)), list(range(91,101)), list(range(101,111)), list(range(111,121)), list(range(121,131)), list(range(131,141))]

    fdr_mu_bh = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fdr_sigma_bh = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fnr_mu_bh = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fnr_sigma_bh = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    atp_mu_bh = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    atp_sigma_bh = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    fdr_mu_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fdr_sigma_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fnr_mu_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fnr_sigma_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    atp_mu_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    atp_sigma_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    fdr_mu_lis_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fdr_sigma_lis_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fnr_mu_lis_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fnr_sigma_lis_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    atp_mu_lis_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    atp_sigma_lis_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    fdr_mu_lis_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fdr_sigma_lis_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fnr_mu_lis_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fnr_sigma_lis_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    atp_mu_lis_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    atp_sigma_lis_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    fdr_sigma_groundtruth = [0.1126, 0.101, 0.086, 0.086, 0.106, 0.100, 0.111]
    fnr_sigma_groundtruth = [0.179, 0.17, 0.16, 0.15, 0.146, 0.1452, 0.136]
    atp_sigma_groundtruth = [315, 363, 754, 1091, 1328, 1386, 1627]
    
    fdr_mu_groundtruth = [0.112, 0.099, 0.095, 0.103, 0.092, 0.094, 0.108]
    fnr_mu_groundtruth = [0.08, 0.084, 0.107, 0.127, 0.151, 0.167, 0.18]
    atp_mu_groundtruth = [3280, 2948, 2476, 1796, 1124, 501, 203]

    for i in range(len(sigma_files)):
        for j in range(len(sigma_files[i])):
            with open(os.path.join(datapath, str(sigma_files[i][j]) + '/result/bh.txt'), "r+") as f:
                fdr_sigma_bh[i] += float(f.readline().split('\n')[0].split(' ')[1])
                fnr_sigma_bh[i] += float(f.readline().split('\n')[0].split(' ')[1])
                atp_sigma_bh[i] += float(f.readline().split('\n')[0].split(' ')[1])

            with open(os.path.join(datapath, str(sigma_files[i][j]) + '/result/qval.txt'), "r+") as f:
                fdr_sigma_q[i] += float(f.readline().split('\n')[0].split(' ')[1])
                fnr_sigma_q[i] += float(f.readline().split('\n')[0].split(' ')[1])
                atp_sigma_q[i] += float(f.readline().split('\n')[0].split(' ')[1])

            with open(os.path.join(datapath, str(sigma_files[i][j]) + '/result/signal.txt'), "r+") as f:
                fdr_sigma_lis_1[i] += float(f.readline().split('\n')[0].split(' ')[1])
                fnr_sigma_lis_1[i] += float(f.readline().split('\n')[0].split(' ')[1])
                atp_sigma_lis_1[i] += float(f.readline().split('\n')[0].split(' ')[1])

            with open(os.path.join(datapath, str(sigma_files[i][j]) + '/result/lis.txt'), "r+") as f:
                fdr_sigma_lis_2[i] += float(f.readline().split('\n')[0].split(' ')[1])
                fnr_sigma_lis_2[i] += float(f.readline().split('\n')[0].split(' ')[1])
                atp_sigma_lis_2[i] += float(f.readline().split('\n')[0].split(' ')[1])

        fdr_sigma_bh[i] /= len(sigma_files[i])
        fnr_sigma_bh[i] /= len(sigma_files[i])
        atp_sigma_bh[i] /= len(sigma_files[i])

        fdr_sigma_q[i] /= len(sigma_files[i])
        fnr_sigma_q[i] /= len(sigma_files[i])
        atp_sigma_q[i] /= len(sigma_files[i])

        fdr_sigma_lis_1[i] /= len(sigma_files[i])
        fnr_sigma_lis_1[i] /= len(sigma_files[i])
        atp_sigma_lis_1[i] /= len(sigma_files[i])

        fdr_sigma_lis_2[i] /= len(sigma_files[i])
        fnr_sigma_lis_2[i] /= len(sigma_files[i])
        atp_sigma_lis_2[i] /= len(sigma_files[i])

    for i in range(len(mu_files)):
        for j in range(len(mu_files[i])):
            with open(os.path.join(datapath, str(mu_files[i][j]) + '/result/bh.txt'), "r+") as f:
                fdr_mu_bh[i] += float(f.readline().split('\n')[0].split(' ')[1])
                fnr_mu_bh[i] += float(f.readline().split('\n')[0].split(' ')[1])
                atp_mu_bh[i] += float(f.readline().split('\n')[0].split(' ')[1])

            with open(os.path.join(datapath, str(mu_files[i][j]) + '/result/qval.txt'), "r+") as f:
                fdr_mu_q[i] += float(f.readline().split('\n')[0].split(' ')[1])
                fnr_mu_q[i] += float(f.readline().split('\n')[0].split(' ')[1])
                atp_mu_q[i] += float(f.readline().split('\n')[0].split(' ')[1])

            with open(os.path.join(datapath, str(mu_files[i][j]) + '/result/signal.txt'), "r+") as f:
                fdr_mu_lis_1[i] += float(f.readline().split('\n')[0].split(' ')[1])
                fnr_mu_lis_1[i] += float(f.readline().split('\n')[0].split(' ')[1])
                atp_mu_lis_1[i] += float(f.readline().split('\n')[0].split(' ')[1])

            with open(os.path.join(datapath, str(mu_files[i][j]) + '/result/lis.txt'), "r+") as f:
                fdr_mu_lis_2[i] += float(f.readline().split('\n')[0].split(' ')[1])
                fnr_mu_lis_2[i] += float(f.readline().split('\n')[0].split(' ')[1])
                atp_mu_lis_2[i] += float(f.readline().split('\n')[0].split(' ')[1])
                
        fdr_mu_bh[i] /= len(sigma_files[i])
        fnr_mu_bh[i] /= len(sigma_files[i])
        atp_mu_bh[i] /= len(sigma_files[i])

        fdr_mu_q[i] /= len(sigma_files[i])
        fnr_mu_q[i] /= len(sigma_files[i])
        atp_mu_q[i] /= len(sigma_files[i])

        fdr_mu_lis_1[i] /= len(sigma_files[i])
        fnr_mu_lis_1[i] /= len(sigma_files[i])
        atp_mu_lis_1[i] /= len(sigma_files[i])

        fdr_mu_lis_2[i] /= len(sigma_files[i])
        fnr_mu_lis_2[i] /= len(sigma_files[i])
        atp_mu_lis_2[i] /= len(sigma_files[i])
    '''

    fdr_mu_bh = [0.077, 0.0757, 0.0875, 0.0858, 0.082, 0.1056, 0.0968]
    fnr_mu_bh = [0.08,  0.091, 0.109, 0.13, 0.155, 0.171, 0.175]
    atp_mu_bh = [3169, 2902, 2241, 1608, 948, 457, 317]

    fdr_sigma_bh = [0.118, 0.1, 0.095, 0.074, 0.0857, 0.091, 0.064]
    fnr_sigma_bh = [0.173, 0.174, 0.167, 0.152, 0.15, 0.14, 0.1366]
    atp_sigma_bh = [252, 261, 436, 988, 1152, 1358, 1439]

    fdr_mu_q = [0.096, 0.0984, 0.1025, 0.098, 0.098, 0.121, 0.094]
    fnr_mu_q = [0.076, 0.0844, 0.104, 0.1247, 0.1504, 0.1685, 0.173]
    atp_mu_q = [3268, 3087, 2406, 1776, 1096, 551, 375]

    fdr_sigma_q = [0.121, 0.128, 0.110, 0.090, 0.1003, 0.107, 0.077]
    fnr_sigma_q = [0.174, 0.171, 0.16, 0.147, 0.1467, 0.137, 0.1339]
    atp_sigma_q = [272, 359, 645, 1154, 1273, 1454, 1522]

    fdr_mu_lis_2 = [0.097, 0.100, 0.11, 0.100, 0.100, 0.11, 0.093]
    fnr_mu_lis_2 = [0.07, 0.0838, 0.102, 0.123, 0.1503, 0.163, 0.166]
    atp_mu_lis_2 = [3387, 3102, 2458, 1818, 1104, 702, 601]

    fdr_sigma_lis_2 = [0.098, 0.093, 0.11, 0.100, 0.11, 0.096, 0.089]
    fnr_sigma_lis_2 = [0.164, 0.164, 0.159, 0.1448, 0.142, 0.1317, 0.1189]
    atp_sigma_lis_2 = [518, 591, 703, 1229, 1399, 1619, 1965]

    fdr_sigma_groundtruth = [0.098,0.093, 0.11, 0.103, 0.112, 0.0955, 0.092]
    fnr_sigma_groundtruth = [0.163,0.164, 0.159, 0.145, 0.1427, 0.1317, 0.119]
    atp_sigma_groundtruth = [520,592, 702, 1223, 1398, 1619, 1961]
    
    fdr_mu_groundtruth = [0.0966, 0.101, 0.112, 0.099, 0.100, 0.11, 0.092]
    fnr_mu_groundtruth = [0.071, 0.0839, 0.102, 0.123, 0.1503, 0.162, 0.166]
    atp_mu_groundtruth = [3391, 3099, 2458, 1821, 1105, 705, 602]
    

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axs[0][0].plot(x_sigma, fdr_sigma_bh, color='r', label='bh', marker='o', markersize=6, linestyle='dashed', mfc='none')
    axs[0][0].plot(x_sigma, fdr_sigma_q, color='b', label='q', marker='D', markersize=6, linestyle='dashed', mfc='none')
    #axs[0][0].plot(x_sigma, fdr_sigma_lis_1, color='y', label='m1', marker='+', markersize=6, linestyle='dashed', mfc='none')
    axs[0][0].plot(x_sigma, fdr_sigma_lis_2, color='g', label='m2', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[0][0].plot(x_sigma, fdr_sigma_groundtruth, color='k', label='gt', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[0][0].legend()
    axs[0][0].set(xlabel=r'$\sigma_1^2$', ylabel='FDR')

    axs[0][1].plot(x_sigma, fnr_sigma_bh, color='r', label='bh', marker='o', markersize=6, linestyle='dashed', mfc='none')
    axs[0][1].plot(x_sigma, fnr_sigma_q, color='b', label='q', marker='D', markersize=6, linestyle='dashed', mfc='none')
    #axs[0][1].plot(x_sigma, fnr_sigma_lis_1, color='y', label='m1', marker='+', markersize=6, linestyle='dashed', mfc='none')
    axs[0][1].plot(x_sigma, fnr_sigma_lis_2, color='g', label='m2', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[0][1].plot(x_sigma, fnr_sigma_groundtruth, color='k', label='gt', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[0][1].legend()
    axs[0][1].set(xlabel=r'$\sigma_1^2$', ylabel='FNR')

    axs[0][2].plot(x_sigma, atp_sigma_bh, color='r', label='bh', marker='o', markersize=6, linestyle='dashed', mfc='none')
    axs[0][2].plot(x_sigma, atp_sigma_q, color='b', label='q', marker='D', markersize=6, linestyle='dashed', mfc='none')
    #axs[0][2].plot(x_sigma, atp_sigma_lis_1, color='y', label='m1', marker='+', markersize=6, linestyle='dashed', mfc='none')
    axs[0][2].plot(x_sigma, atp_sigma_lis_2, color='g', label='m2', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[0][2].plot(x_sigma, atp_sigma_groundtruth, color='k', label='gt', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[0][2].legend()
    axs[0][2].set(xlabel=r'$\sigma_1^2$', ylabel='ATP')

    axs[1][0].plot(x_mu, fdr_mu_bh, color='r', label='bh', marker='o', markersize=6, linestyle='dashed', mfc='none')
    axs[1][0].plot(x_mu, fdr_mu_q, color='b', label='q', marker='D', markersize=6, linestyle='dashed', mfc='none')
    #axs[1][0].plot(x_mu, fdr_mu_lis_1, color='y', label='m1', marker='+', markersize=6, linestyle='dashed', mfc='none')
    axs[1][0].plot(x_mu, fdr_mu_lis_2, color='g', label='m2', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[1][0].plot(x_mu, fdr_mu_groundtruth, color='k', label='gt', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[1][0].legend()
    axs[1][0].set(xlabel=r'$\mu_1$', ylabel='FDR')

    axs[1][1].plot(x_mu, fnr_mu_bh, color='r', label='bh', marker='o', markersize=6, linestyle='dashed', mfc='none')
    axs[1][1].plot(x_mu, fnr_mu_q, color='b', label='q', marker='D', markersize=6, linestyle='dashed', mfc='none')
    #axs[1][1].plot(x_mu, fnr_mu_lis_1, color='y', label='m1', marker='+', markersize=6, linestyle='dashed', mfc='none')
    axs[1][1].plot(x_mu, fnr_mu_lis_2, color='g', label='m2', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[1][1].plot(x_mu, fnr_mu_groundtruth, color='k', label='gt', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[1][1].legend()
    axs[1][1].set(xlabel=r'$\mu_1$', ylabel='FNR')

    axs[1][2].plot(x_mu, atp_mu_bh, color='r', label='bh', marker='o', markersize=6, linestyle='dashed', mfc='none')
    axs[1][2].plot(x_mu, atp_mu_q, color='b', label='q', marker='D', markersize=6, linestyle='dashed', mfc='none')
    #axs[1][2].plot(x_mu, atp_mu_lis_1, color='y', label='m1', marker='+', markersize=6, linestyle='dashed', mfc='none')
    axs[1][2].plot(x_mu, atp_mu_lis_2, color='g', label='m2', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[1][2].plot(x_mu, atp_mu_groundtruth, color='k', label='gt', marker='s', markersize=6, linestyle='dashed', mfc='none')
    axs[1][2].legend()
    axs[1][2].set(xlabel=r'$\mu_1$', ylabel='ATP')
    plt.tight_layout()
    plt.show()





