# import collections

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy 
from statsmodels.stats.multitest import local_fdr, NullDistribution
import os 
import time
np.bool = np.bool_

class EarlyStop:
    def __init__(self, patience: int = 3, threshold: float = 1e-2) -> None:
        # self.queue = collections.deque([0] * patience, maxlen=patience)
        self.patience = patience
        self.threshold = threshold
        self.wait = 0
        self.best_loss = np.Inf

    def __call__(self, train_loss: float) -> bool:
        """
        @monitor: value to monitor for early stopping
                  (e.g. train_loss, test_loss, ...)
        @mode: specify whether you want to maximize or minimize
               relative to @monitor
        """
        if np.less(self.threshold, 0):
            return False
        if train_loss is None:
            return False

        if np.less(train_loss - self.best_loss, -self.threshold):
            self.best_loss = train_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False

class FDR_MSELoss(nn.Module):
    def __init__(self):
        super(FDR_MSELoss, self).__init__()
        self.lambda_reg = 0.0

    def forward(self, p_value, predicted_p_value):
        # Calculate the Mean Squared Error (MSE) loss between true and predicted p-values
        mse_loss = torch.mean((predicted_p_value - p_value)**2)

        # Regularization term for FDR control
        #penalty = torch.mean(torch.abs(predicted_p_value - 0.5)) # sometimes the model gets stuck, generating predictions near 0.5

        # Combine the MSE loss and regularization term
        total_loss = mse_loss 

        return total_loss

class SoftNCutLoss3D(nn.Module):
    def __init__(self):
        super(SoftNCutLoss3D, self).__init__()

    def calculate_weights(self, input, batch_size, img_size=(30, 30, 30), ox=3, radius=3 ,oi=11):
        channels = 1

        d, h, w = img_size
        p = radius

        image = torch.mean(input, dim=1, keepdim=True)

        image = F.pad(input=image, pad=(p, p, p, p, p, p), mode='constant', value=0)

        kd, kh, kw = radius * 2 + 1, radius * 2 + 1, radius * 2 + 1
        dd, dh, dw = 1, 1, 1

        patches = image.unfold(2, kd, dd).unfold(3, kh, dh).unfold(4, kw, dw)

        patches = patches.contiguous().view(batch_size, channels, -1, kd, kh, kw)

        patches = patches.permute(0, 2, 1, 3, 4, 5)
        patches = patches.view(-1, channels, kd, kh, kw)

        center_values = patches[:, :, radius, radius, radius]
        center_values = center_values[:, :, None, None, None]
        center_values = center_values.expand(-1, -1, kd, kh, kw)

        k_row = (torch.arange(1, kd + 1) - torch.arange(1, kd + 1)[radius]).expand(kd, kh, kw)

        if torch.cuda.is_available():
            k_row = k_row.cuda()

        distance_weights = (k_row ** 2 + k_row.unsqueeze(-1) ** 2 + k_row.unsqueeze(-2) ** 2)

        mask = distance_weights.le(radius)
        distance_weights = torch.exp(-distance_weights / (ox**2))
        distance_weights = torch.mul(mask, distance_weights)

        patches = torch.exp(-((patches - center_values)**2) / (oi**2))
        return torch.mul(patches, distance_weights)

    def soft_n_cut_loss_single_k(self, weights, enc, batch_size, img_size=(30, 30, 30), radius=3):
        channels = 1
        d, h, w = img_size
        p = radius

        kd, kh, kw = radius * 2 + 1, radius * 2 + 1, radius * 2 + 1
        dd, dh, dw = 1, 1, 1

        enc = enc.unsqueeze(0)
        encoding = F.pad(input=enc, pad=(p, p, p, p, p, p), mode='constant', value=0)

        seg = encoding.unfold(2, kd, dd).unfold(3, kh, dh).unfold(4, kw, dw)
        seg = seg.contiguous().view(batch_size, channels, -1, kd, kh, kw)

        seg = seg.permute(0, 2, 1, 3, 4, 5)
        seg = seg.view(-1, channels, kd, kh, kw)

        nom = weights * seg

        nominator = torch.sum(enc * torch.sum(nom, dim=(1, 2, 3, 4)).reshape(batch_size, d, h, w), dim=(1, 2, 3, 4))
        denominator = torch.sum(enc * torch.sum(weights, dim=(1, 2, 3, 4)).reshape(batch_size, d, h, w), dim=(1, 2, 3, 4))

        return torch.div(nominator, denominator)

    def forward(self, image, enc):
        batch_size = 1
        k = 2
        weights = self.calculate_weights(image, batch_size)
        
        loss = []
        for i in range(k):
            if i == 0: loss.append(self.soft_n_cut_loss_single_k(weights, enc[:, 0], batch_size))
            elif i == 1: loss.append(self.soft_n_cut_loss_single_k(weights, 1-enc[:, 0], batch_size))
        
        da = torch.stack(loss)
        loss = torch.mean(k - torch.sum(da, dim=0)) 

        #penalty = torch.mean(torch.abs(enc - 0.5)) # sometimes the model gets stuck, generating predictions near 0.5

        return loss 

def dice(true_mask, pred_mask, non_seg_score=1.0):
    """
        Computes the Dice coefficient.
        Args:
            true_mask : Array of arbitrary shape.
            pred_mask : Array with the same shape than true_mask.  
        
        Returns:
            A scalar representing the Dice coefficient between the two segmentations. 
        
    """
    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(np.bool)
    pred_mask = np.asarray(pred_mask).astype(np.bool)

    # If both segmentations are all zero, the dice will be 1. (Developer decision)
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    # Compute Dice coefficient
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum

def qvalue(pvals, threshold=0.05, verbose=False):
    """Function for estimating q-values from p-values using the Storey-
    Tibshirani q-value method (2003).

    Input arguments:
    ================
    pvals       - P-values corresponding to a family of hypotheses.
    threshold   - Threshold for deciding which q-values are significant.

    Output arguments:
    =================
    significant - An array of flags indicating which p-values are significant.
    qvals       - Q-values corresponding to the p-values.
    """

    """Count the p-values. Find indices for sorting the p-values into
    ascending order and for reversing the order back to original."""
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    rev_ind = np.argsort(ind)
    pvals = pvals[ind]

    # Estimate proportion of features that are truly null.
    kappa = np.arange(0, 0.96, 0.01)
    pik = [sum(pvals > k) / (m*(1-k)) for k in kappa]
    cs = UnivariateSpline(kappa, pik, k=3, s=None, ext=0)
    pi0 = float(cs(1.))
    if (verbose):
        print('The estimated proportion of truly null features is %.3f' % pi0)

    """The smoothing step can sometimes converge outside the interval [0, 1].
    This was noted in the published literature at least by Reiss and
    colleagues [4]. There are at least two approaches one could use to
    attempt to fix the issue:
    (1) Set the estimate to 1 if it is outside the interval, which is the
        assumption in the classic FDR method.
    (2) Assume that if pi0 > 1, it was overestimated, and if pi0 < 0, it
        was underestimated. Set to 0 or 1 depending on which case occurs.

    I'm choosing second option 
    """
    if pi0 < 0:
        pi0 = 0
    elif pi0 > 1:
        pi0 = 1

    # Compute the q-values.
    qvals = np.zeros(np.shape(pvals))
    qvals[-1] = pi0*pvals[-1]
    for i in np.arange(m-2, -1, -1):
        qvals[i] = min(pi0*m*pvals[i]/float(i+1), qvals[i+1])

    # Test which p-values are significant.
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind] = qvals<threshold

    """Order the q-values according to the original order of the p-values."""
    qvals = qvals[rev_ind]
    return significant, qvals

def compute_qval(x, alpha):
    # calculate p_value from z-score

    x_ = np.ravel(x.copy())
    p_value = scipy.stats.norm.sf(np.fabs(x_))*2.0 # two-sided tail, calculates 1-cdf
    p_value = np.ravel(p_value)
    qv = qvalue(p_value, threshold=alpha)[1]
    return qv

def compute_bh(x, alpha):
    # calculate p_value from z-score
    x_ = np.ravel(x.copy())
    p_value = scipy.stats.norm.sf(np.fabs(x_))*2.0 # two-sided tail, calculates 1-cdf
    p_value = np.ravel(p_value)    
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_value, alpha=alpha, method='fdr_bh')
    return reject, pvals_corrected, alphacSidak, alphacBonf

def p_lis(gamma_1, threshold=0.1, label=None, savepath=None, flip=False):
    '''
    Rejection of null hypothesis are shown as 1, consistent with online BH, Q-value, smoothFDR methods.
    # LIS = P(theta = 0 | x)
    # gamma_1 = P(theta = 1 | x) = 1 - LIS
    '''
    gamma_1 = gamma_1.ravel()
    dtype = [('index', int), ('value', float)]
    size = gamma_1.shape[0]

    # flip
    lis = np.zeros(size, dtype=dtype)
    lis[:]['index'] = np.arange(0, size)
    lis[:]['value'] = 1-gamma_1 if flip else gamma_1

    # get k
    lis = np.sort(lis, order='value')
    cumulative_sum = np.cumsum(lis[:]['value'])
    k = np.argmax(cumulative_sum > (np.arange(len(lis)) + 1)*threshold)

    signal_lis = np.zeros(size)
    signal_lis[lis[:k]['index']] = 1

    if savepath is not None:
        np.save(savepath + 'gamma.npy', gamma_1)
        np.save(savepath + 'lis.npy', signal_lis)

    if label is not None:
        # GT FDP
        rx = k
        sigx = np.sum(1-label[lis[:k]['index']])
        fdr = sigx / rx if rx > 0 else 0

        # GT FNR
        rx = size - k
        sigx = np.sum(label[lis[k:]['index']]) 
        fnr = sigx / rx if rx > 0 else 0

        # GT ATP
        atp = np.sum(label[lis[:k]['index']]) 
        return fdr, fnr, atp
    return k, lis # contains index, value of the roi lis


def lfdr(x, alpha, null):
    """
    https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.local_fdr.html
    """
    z_score = np.ravel(x.copy())
    print(null.null_proportion)
    fdr = local_fdr(z_score, null_proportion=null.null_proportion) 
    lfdr_signals = np.zeros(z_score.shape[0])

    for i in range(z_score.shape[0]):
        if fdr[i] <= alpha:
            lfdr_signals[i] = 1
    return lfdr_signals





