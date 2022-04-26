import matplotlib.pylab as plt
import numpy as np
from smoothfdr.easy import smooth_fdr
from smoothfdr.utils import *
import os
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy import interpolate
import scipy.stats


root = '/scratch/tk2737/ADNI/'


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


def compute_bh_and_qval(x, label):
    # calculate p_value from z-score
    p_value = scipy.stats.norm.sf(np.fabs(x))*2.0 # two-sided tail, calculates 1-cdf
    p_value = np.ravel(p_value)

    ############################ BH method ##############################################
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_value, alpha=0.1, method='fdr_bh')

    ############################ Q-value method ##############################################

    qv = q_estimate(p_value)

    reject = reject

    num_rejected = np.count_nonzero(reject)
    num_not_rejected = (x.shape[0]) - num_rejected
    fdr = 0
    fnr = 0
    atp = 0

    fdr_q = 0
    fnr_q = 0
    atp_q = 0
    num_rejected_q = 0
    for i in range(x.shape[0]):
        if reject[i] == 1:  # rejected
            if label[i] == 0:
                fdr += 1
            elif label[i] == 1:
                atp += 1
        elif reject[i] == 0:  # not rejected
            if label[i] == 1:
                fnr += 1
        if qv[i] < 0.1: # reject
            num_rejected_q += 1
            if label[i] == 0:
                fdr_q += 1
            elif label[i] == 1:
                atp_q += 1
        else: # not rejected
            if label[i] == 1:
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

    if (x.shape[0]) - num_rejected_q == 0:
        fnr_q = 0
    else:
        fnr_q /= (x.shape[0]) - num_rejected_q

    return fdr, fnr, atp, fdr_q, fnr_q, atp_q

def run_27000_simulation():
    savepath_list = []
    labelpath_list = []
    #cube_list = ['01003', '01512', '02011', '02496', '02952', '005033']
    cube_list = ['02011', '02496', '02952', '005033']
    mu_list = ['mu_n4_2', 'mu_n35_2', 'mu_n3_2', 'mu_n25_2', 'mu_n2_2', 'mu_n15_2', 'mu_n1_2']
    sigma_list = ['sigma_125_1', 'sigma_25_1', 'sigma_5_1', 'sigma_1_1', 'sigma_2_1', 'sigma_4_1', 'sigma_8_1']

    root = '/Users/taehyo/Dropbox/NYU/Research/Research/Code/data/adni_simulation/adni_good/'

    for cube in cube_list:
        for mu in mu_list:
            path = root + cube + '/' + mu 
            label = root + cube + '/label/label.npy'
            savepath_list.append(path)
            labelpath_list.append(label)

        for sig in sigma_list:
            path = root + cube + '/' + sig 
            label = root + cube + '/label/label.npy'
            savepath_list.append(path)
            labelpath_list.append(label)

    for index in range(len(savepath_list)):
        print(savepath_list[index])
        data = np.ravel(np.load(savepath_list[index] + '/x_1d.npy'))
        label = np.ravel(np.load(labelpath_list[index]))

        fdr_level = 0.1

        # Runs the FDR smoothing algorithm with the default settings
        # and using a 2d grid edge set
        # Note that verbose is a level not a boolean, so you can
        # set verbose=0 for silent, 1 for high-level output, 2+ for more details

        
        results = smooth_fdr(data, fdr_level, verbose=1, missing_val=0)
        predicted_signals = np.ravel(results['discoveries'])

        num_rejected = np.count_nonzero(predicted_signals)

        num_not_rejected = (predicted_signals.shape[0]) - num_rejected
        fdr = 0
        fnr = 0
        atp = 0

        for i in range(predicted_signals.shape[0]):
            if predicted_signals[i] == 1: # rejected
                if label[i] == 0:
                    fdr += 1
                elif label[i] == 1:
                    atp += 1
            elif predicted_signals[i] == 0: # not rejected
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

        # Save final signal_file

        with open(os.path.join(savepath_list[index], 'result/smooth.txt'), 'w') as outfile:
            outfile.write('smoothFDR:\n')
            outfile.write('fdr: ' + str(fdr) + '\n')
            outfile.write('fnr: ' + str(fnr) + '\n')
            outfile.write('atp: ' + str(atp) + '\n')
     

def get_roi_map():
    # get ROI region, set background voxels as 0
    # extract cropped image 
    roi = nib.load(os.path.join(root, 'labels_Neuromorphometrics.nii'))
    roi_img = np.array(roi.dataobj)

    rois = set()
    with open(os.path.join(root, 'spm_templates.man'), 'r',  encoding="utf8") as file:
        for i, line in enumerate(file):
            if i >= 87 and i <= 222:
                roi_label = int(line.split()[1])
                rois.add(roi_label)


    exclude = {4, 11, 40, 41, 44, 45, 63, 64, 46, 51, 52, 61, 62, 69, 49, 50}
    rois = rois - exclude
    for i in range(roi_img.shape[0]):
        for j in range(roi_img.shape[1]):
            for k in range(roi_img.shape[2]):
                if roi_img[i,j,k] in rois:
                    roi_img[i,j,k] = 1
                else:
                    roi_img[i,j,k] = 0

    mins, maxes, count = bound(roi_img)
    maxes = maxes.astype(int)
    roi_img = roi_img[mins[0]:maxes[0]+1, mins[1]:maxes[1]+1, mins[2]:maxes[2]+1]

    return roi_img

def get_adni_status():
    # get list of EMCI and CN files
    index_AD = []
    index_CN = []
    with open(os.path.join(root, 'ADNI_FDGPET_subjectID_1536_status.txt'), 'r') as file:
        for i, line in enumerate(file):
            if line.split()[1] == 'EMCI':
                index_AD.append(i-1)
            elif line.split()[1] == 'CN':
                index_CN.append(i-1)
    return index_AD, index_CN

if __name__ == "__main__":
    # Code for Real Data 
    data = np.ravel(np.load(root + 'x_1d.npy'))

    fdr_level = 1e-3

    # Runs the FDR smoothing algorithm with the default settings
    # and using a 2d grid edge set
    # Note that verbose is a level not a boolean, so you can
    # set verbose=0 for silent, 1 for high-level output, 2+ for more details

    results = smooth_fdr(data, fdr_level, verbose=1, missing_val=0)
    predicted_signals = np.ravel(results['discoveries'])

    if not os.path.isdir(root + 'result/'):
        os.makedirs(root + 'result/')

    with open(os.path.join(root, 'result/smooth.txt'), 'w') as outfile:
        outfile.write('smoothFDR:\n')
        outfile.write('signals: ' + str(np.count_nonzero(predicted_signals)) + '\n')

    # convert 1d to 3d, and map the signals to a brain image
    roi_map = get_roi_map()
    index_AD, index_CN = get_adni_status()
    roi_img = np.memmap(os.path.join(root, 'EMCI', 'EMCI.npy'),  dtype='float64', mode='r', shape=(len(index_AD), roi_map.shape[0], roi_map.shape[1], roi_map.shape[2]))[0] 

    nd_signals = np.zeros(roi_map.shape)
    index = 0
    for i in range(roi_map.shape[0]):
        for j in range(roi_map.shape[1]):
            for k in range(roi_map.shape[2]):
                if roi_map[i,j,k]:
                    nd_signals[i,j,k] = predicted_signals[index]
                    index += 1

    cropped = nib.Nifti1Image(nd_signals*roi_img, np.eye(4))
    cropped.header.get_xyzt_units()
    cropped.to_filename(root + 'result/smooth.nii')