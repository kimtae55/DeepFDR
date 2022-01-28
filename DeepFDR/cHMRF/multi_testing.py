
# 1. calculate p value from z-test statistics
# 2. run BH-method from p value
# 3. run Q-value method from p value
# generate gamma_file using the output probability
# calculate fdr, fnr, atp

import scipy.stats
import numpy as np
from test_data_cpu import Data
import sys
from statsmodels.sandbox.stats.multicomp import multipletests
import os

import scipy as sp
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


if __name__ == "__main__":
    rng_seed = sys.argv[1]
    print("RAND: ", rng_seed)
    np.random.seed(int(rng_seed))

    label_file = './label.txt'
    label = np.loadtxt(label_file).reshape((Data.SIZE, Data.SIZE, Data.SIZE))
    x_file = './x_val_direct.txt'
    savepath = './'
    # z-score
    x = np.loadtxt(x_file).reshape((Data.SIZE, Data.SIZE, Data.SIZE))
    # calculate p_value from z-score
    p_value = scipy.stats.norm.sf(np.fabs(x))*2.0 # two-sided tail, calculates 1-cdf
    p_value = np.ravel(p_value)

    ############################ BH method ##############################################
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_value, alpha=0.1, method='fdr_bh')
    print(np.amin(pvals_corrected))
    print("reject", reject)
    print("pvals_corrected", pvals_corrected)
    print("alphacSidak", alphacSidak)
    print("alphacBonf", alphacBonf)

    ############################ Q-value method ##############################################
    qv = q_estimate(p_value).reshape((Data.SIZE, Data.SIZE, Data.SIZE))

    reject = reject.reshape((Data.SIZE, Data.SIZE, Data.SIZE))
    print("num_rejected", np.count_nonzero(reject))

    num_rejected = np.count_nonzero(reject)
    num_not_rejected = (Data.SIZE * Data.SIZE * Data.SIZE) - num_rejected
    fdr = 0
    fnr = 0
    atp = 0

    fdr_q = 0
    fnr_q = 0
    atp_q = 0
    num_rejected_q = 0
    for i in range(Data.SIZE):
        for j in range(Data.SIZE):
            for k in range(Data.SIZE):
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

    if (Data.SIZE * Data.SIZE * Data.SIZE) - num_rejected_q == 0:
        fnr_q = 0
    else:
        fnr_q /= (Data.SIZE * Data.SIZE * Data.SIZE) - num_rejected_q

    # Save final signal_file
    signal_directory = savepath + 'bh.txt'
    with open(signal_directory, 'w') as outfile:
        outfile.write('fdr: ' + str(fdr) + '\n')
        outfile.write('fnr: ' + str(fnr) + '\n')
        outfile.write('atp: ' + str(atp) + '\n')
        for data_slice in reject:
            np.savetxt(outfile, data_slice, fmt='%-8.4f')
            outfile.write('# New z slice\n')

    # Save final signal_file
    signal_directory = savepath + 'qval.txt'
    with open(signal_directory, 'w') as outfile:
        outfile.write('fdr: ' + str(fdr_q) + '\n')
        outfile.write('fnr: ' + str(fnr_q) + '\n')
        outfile.write('atp: ' + str(atp_q) + '\n')
        for data_slice in qv:
            np.savetxt(outfile, data_slice, fmt='%-8.4f')
            outfile.write('# New z slice\n')