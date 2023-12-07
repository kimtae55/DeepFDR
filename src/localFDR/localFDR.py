import numpy as np 
from statsmodels.stats.multitest import local_fdr, NullDistribution 
import os 
import time 
from ..DeepFDR import util

def lfdr(x, alpha, null):
    """
    https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.local_fdr.html
    """
    z_score = np.ravel(x.copy())
    fdr = local_fdr(z_score, null_proportion=null.null_proportion) 
    lfdr_signals = np.zeros(z_score.shape[0])

    for i in range(z_score.shape[0]):
        if fdr[i] <= alpha:
            lfdr_signals[i] = 1
    return lfdr_signals

if __name__ == "__main__":
    savepath_list = []
    mu_list = ['mu_n05_2', 'mu_n0_2'] #'mu_n4_2', 'mu_n35_2', 'mu_n3_2', 'mu_n25_2', 'mu_n2_2', 'mu_n15_2', 'mu_n1_2',
    sigma_list = [] # 'sigma_125_1', 'sigma_25_1', 'sigma_5_1', 'sigma_1_1', 'sigma_2_1', 'sigma_4_1', 'sigma_8_1'

    root = '/Users/taehyo/Dropbox/NYU/Research/Research/Data/deepfdr/data/'

    alpha = 0.1
    for sig in sigma_list:
        path = root + '/' + 'sigma/' + sig
        savepath_list.append(path)

    for mu in mu_list:
        path = root + '/' + 'mu/' + mu
        savepath_list.append(path)

    times_lfdr = []
    times_bh = []
    times_qval = []

    mu_fdrs = []
    sigma_fdrs = []
    mu_fnrs = []
    sigma_fnrs = []
    mu_atps = []
    sigma_atps = []

    bh_mu_fdrs = []
    bh_sigma_fdrs = []
    bh_mu_fnrs = []
    bh_sigma_fnrs = []
    bh_mu_atps = []
    bh_sigma_atps = []

    qv_mu_fdrs = []
    qv_sigma_fdrs = []
    qv_mu_fnrs = []
    qv_sigma_fnrs = []
    qv_mu_atps = []
    qv_sigma_atps = []

    for index in range(len(savepath_list)):
        print('------------------------------------------------------------')
        print('-----------------       ', savepath_list[index])
        print('------------------------------------------------------------')
        lfdrs = []
        lfnrs = []
        latps = []

        bh_fdrs = []
        bh_fnrs = []
        bh_atps = []

        qv_fdrs = []
        qv_fnrs = []
        qv_atps = []
        for i in range(10):
            try:
                # LFDR
                start = time.time()
                data = np.load(os.path.join(savepath_list[index]+'/data0.1.npy'))[i]
                label = np.ravel(np.load(root+'/cubes0.1.npy')[0])
                null = NullDistribution(data, estimate_mean=False, estimate_scale=False, estimate_null_proportion=True)
                predicted_signals = lfdr(data,alpha, null)

                num_rejected = np.count_nonzero(predicted_signals)
                num_not_rejected = (predicted_signals.shape[0]) - num_rejected
                fdr = sum(1 for i in range(len(predicted_signals)) if predicted_signals[i] == 1 and label[i] == 0) / max(num_rejected, 1)
                fnr = sum(1 for i in range(len(predicted_signals)) if predicted_signals[i] == 0 and label[i] == 1) / max(num_not_rejected, 1)
                atp = sum(1 for i in range(len(predicted_signals)) if predicted_signals[i] == 1 and label[i] == 1) 

                end = time.time()
                times_lfdr.append(end-start)

                # Save final signal_file
                lfdrs.append(fdr)
                lfnrs.append(fnr)
                latps.append(atp)

                # BH
                start = time.time()
                bh_signals = util.compute_bh(data, alpha)[0]
                num_rejected = np.count_nonzero(bh_signals)
                num_not_rejected = (bh_signals.shape[0]) - num_rejected
                fdr = sum(1 for i in range(len(bh_signals)) if bh_signals[i] == 1 and label[i] == 0) / max(num_rejected, 1)
                fnr = sum(1 for i in range(len(bh_signals)) if bh_signals[i] == 0 and label[i] == 1) / max(num_not_rejected, 1)
                atp = sum(1 for i in range(len(bh_signals)) if bh_signals[i] == 1 and label[i] == 1) 
                end = time.time()
                times_bh.append(end-start)
                bh_fdrs.append(fdr)
                bh_fnrs.append(fnr)
                bh_atps.append(atp)

                # QVALUE
                start = time.time()
                qval = util.compute_qval(data, alpha)
                qval_signals = np.where(qval <= alpha, 1, 0)
                num_rejected = np.count_nonzero(qval_signals)
                num_not_rejected = (qval_signals.shape[0]) - num_rejected
                fdr = sum(1 for i in range(len(qval_signals)) if qval_signals[i] == 1 and label[i] == 0) / max(num_rejected, 1)
                fnr = sum(1 for i in range(len(qval_signals)) if qval_signals[i] == 0 and label[i] == 1) / max(num_not_rejected, 1)
                atp = sum(1 for i in range(len(qval_signals)) if qval_signals[i] == 1 and label[i] == 1) 
                end = time.time()
                times_qval.append(end-start)
                qv_fdrs.append(fdr)
                qv_fnrs.append(fnr)
                qv_atps.append(atp)
            except Exception as e:
                print(e)
                continue

        # compute mean and std for above 50 samples and sort them into sigma/mu arrays 
        def organize(sig_dest, mu_dest, source):
            mfdr, sfdr = np.mean(source[0]), np.std(source[0])
            mfnr, sfnr = np.mean(source[1]), np.std(source[1])
            matp, satp = np.mean(source[2]), np.std(source[2])

            if index < 7:
                sig_dest[0].append(mfdr)
                sig_dest[0].append(sfdr)
                sig_dest[1].append(mfnr)
                sig_dest[1].append(sfnr)
                sig_dest[2].append(matp)
                sig_dest[2].append(satp)

            else:
                mu_dest[0].append(mfdr)
                mu_dest[0].append(sfdr)
                mu_dest[1].append(mfnr)
                mu_dest[1].append(sfnr)
                mu_dest[2].append(matp)
                mu_dest[2].append(satp)

        organize((sigma_fdrs, sigma_fnrs, sigma_atps),
                 (mu_fdrs, mu_fnrs, mu_atps),
                 (lfdrs, lfnrs, latps))
        organize((bh_sigma_fdrs, bh_sigma_fnrs, bh_sigma_atps),
                 (bh_mu_fdrs, bh_mu_fnrs, bh_mu_atps),
                 (bh_fdrs, bh_fnrs, bh_atps))
        organize((qv_sigma_fdrs, qv_sigma_fnrs, qv_sigma_atps),
                 (qv_mu_fdrs, qv_mu_fnrs, qv_mu_atps),
                 (qv_fdrs, qv_fnrs, qv_atps))

    print('------------------> LFDR:')
    print('SIGMA')
    form = [f'{x:.4f}' for x in sigma_fdrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in sigma_fnrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in sigma_atps]
    result = ', '.join(form)
    print(result)
    print('MU')
    form = [f'{x:.4f}' for x in mu_fdrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in mu_fnrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in mu_atps]
    result = ', '.join(form)
    print(result)
    print('time:')
    print('lfdr: ', np.mean(times_lfdr), np.std(times_lfdr))

    print('------------------> BH:')
    print('SIGMA')
    form = [f'{x:.4f}' for x in bh_sigma_fdrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in bh_sigma_fnrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in bh_sigma_atps]
    result = ', '.join(form)
    print(result)
    print('MU')
    form = [f'{x:.4f}' for x in bh_mu_fdrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in bh_mu_fnrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in bh_mu_atps]
    result = ', '.join(form)
    print(result)   
    print('time:')
    print('bh: ', np.mean(times_bh), np.std(times_bh))

    print('------------------> QVAL:')
    print('SIGMA')
    form = [f'{x:.4f}' for x in qv_sigma_fdrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in qv_sigma_fnrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in qv_sigma_atps]
    result = ', '.join(form)
    print(result)
    print('MU')
    form = [f'{x:.4f}' for x in qv_mu_fdrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in qv_mu_fnrs]
    result = ', '.join(form)
    print(result)
    form = [f'{x:.4f}' for x in qv_mu_atps]
    result = ', '.join(form)
    print(result)
    print('time:')
    print('qval: ', np.mean(times_qval), np.std(times_qval))



