import numpy as np
import os
from test_statistic import Data
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numba
import pandas as pd
from numba import cuda, float32, int32, guvectorize, vectorize, config, float64

class Model2:
    def __init__(self, x_file, rng_seed):
        torch.set_default_dtype(torch.float64)
        self.SAVE_DIR = os.path.join(os.getcwd(), '../data/model2/' + str(rng_seed) +'/result')
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        self.params = torch.tensor([-94.5000, -911.2500, -6302.1609, -6505.2827, -707.8399])
        self.groundtruth_phi = torch.tensor([  -81.0000,  -729.0000, -5251.8007, -5692.1224,  -606.7199]) # groundtruth  for 0.11
        self.params_prev = torch.tensor([-94.5000, -911.2500, -6302.1609, -6505.2827, -707.8399])
        self.pL = torch.tensor([0.5, 0.5])
        self.muL = torch.tensor([-3.0, 2.0])
        self.sigmaL2 = torch.tensor([1.0, 1.0])
        self.pL_prev = torch.zeros(2)
        self.muL_prev = torch.zeros(2)
        self.sigmaL2_prev = torch.zeros(2)

        self.const = {'a': 1,  # for penalized likelihood for L >= 2
                      'b': 2,
                      'delta': 1e-3,
                      'maxIter': 200,
                      'eps1': 1.0,
                      'eps2': 0.2,
                      'eps3': 0.02,
                      'eps4': 1e-3,
                      'eps5': 1e-2,
                      'alpha': 1e-3,
                      'burn_in': 80, # 80
                      'num_samples': 50, # 30
                      'L': 2,
                      'newton_max': 3,
                      'fdr_control': 0.1,
                      }

        self.x = torch.from_numpy(np.loadtxt(x_file).reshape((Data.SIZE,Data.SIZE,Data.SIZE)))
        self.gamma = torch.zeros(self.x.shape)  # P(theta | x)
        self.init = torch.zeros(self.x.shape)  # initial theta value for gibb's sampling
        self.H_x = torch.zeros(5)
        self.H_mean = torch.zeros(5)
        self.log_sum_max_elem = torch.zeros(1)
        self.exp_sum = torch.zeros(self.const['num_samples'])
        self.U = torch.zeros(5)
        self.I_inv = torch.zeros((5, 5))
        self.H_iter = torch.zeros((self.const['num_samples'], 5))
        self.H_x_iter = torch.zeros((self.const['num_samples'], 5))
        self.convergence_count = 0
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        init_prob = torch.full((Data.SIZE, Data.SIZE, Data.SIZE), 0.5)
        self.init = torch.bernoulli(init_prob)  # p = 0.5 to choose 1
        self.H_reparam = torch.zeros(5)
        self.reparameterization_factor()
        self.Q2 = torch.zeros(self.const['maxIter'])

        self.t_ij = np.load(os.path.join(os.getcwd(), 't_ij.npy'))
        self.theta_sum = np.load(os.path.join(os.getcwd(), 'theta_sum.npy'))

        self.start = time.time()
        self.end = 0


    def gem(self):
        '''
        E-step: Compute the following
        phi: (6), (7), (8)
        varphi: (10), (11)
        Iterate until convergence using delta (13)
        '''
        # compute first term in hs(t) = h(t) - .... equation
        mu_0 = 0
        sigma0_sq = 1
        const_0 = 1 / torch.sqrt(torch.tensor([2 * torch.pi * sigma0_sq]))
        ln_0 = const_0 * torch.exp(-(torch.pow(self.x - mu_0, 2)) / (2 * sigma0_sq))

        for t in range(self.const['maxIter']):
            print("")
            print("GEM iter: ", t)
            print("params:  " + str(self.params))
            print("params_prev: " + str(self.params_prev))
            print("pL:  " + str(self.pL) + " pL_prev: " + str(self.pL_prev))
            print("muL:  " + str(self.muL) + " muL_prev: " + str(self.muL_prev))
            print("sigmaL2:  " + str(self.sigmaL2) + " sigmaL2_prev: " + str(self.sigmaL2_prev))
            print("groundtruth_phi: ", self.groundtruth_phi)

            # varphi: (10), (11)
            self.params_prev = self.params.clone()
            self.muL_prev = self.muL.clone()
            self.sigmaL2_prev = self.sigmaL2.clone()
            self.pL_prev = self.pL.clone()

            # calculate hs(t)
            ln_1 = 0
            for l in range(self.const['L']):
                ln_1 += ((self.pL[l]) / (torch.sqrt(2 * torch.pi * self.sigmaL2[l]))) * torch.exp(-(torch.pow(self.x - self.muL[l], 2)) / (2 * self.sigmaL2[l]))
            # ----------------------------------------CHECKPOINT MARKER-------------------------------------------------------#
            # Gibb's sampler to generate theta and theta_x
            self.theta, self.H_mean, self.H_iter, self.exp_sum, g = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], torch.zeros(self.x.shape))
            hs = torch.log(ln_1 / ln_0) 
            theta_x, self.H_x, self.H_x_iter, exp_x_sum, self.gamma = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], hs)

            # Monte carlo averages
            self.H_mean /= self.const['num_samples']
            self.H_x /= self.const['num_samples']
            self.gamma /= self.const['num_samples']
            gamma_sum = torch.sum(self.gamma)
            
            
            # compute log factor for current varphi
            # Because np.exp can overflow with large value of exp_sum, a modification is made
            # where we subtract by max_val. Same is done for log_sum_next
            self.log_sum_max_elem = torch.amax(self.exp_sum)
            # compute U, I
            # U --> 5x1 matrix, I --> 5x5 matrix, trans(U)*inv(I)*U = 1x1 matrix
            self.U = self.H_x - self.H_mean
            self.I_inv = self.compute_I_inverse(self.const['num_samples'], self.H_iter, self.H_mean)
            # phi: (6), (7), (8)

            f_x = torch.zeros(self.x.shape)
            omega_sum = torch.zeros(self.const['L'])
            omega = torch.empty((2, Data.SIZE, Data.SIZE, Data.SIZE))

            for l in range(self.const['L']):
                f_x += self.pL[l] * self.norm(self.x, self.muL[l], self.sigmaL2[l])
            for l in range(self.const['L']):
                omega[l] = self.gamma * self.pL[l] / f_x * self.norm(self.x, self.muL[l], self.sigmaL2[l])
                omega_sum[l] = torch.sum(omega[l])

            print("omega_sum: ", omega_sum)
            print("gamma_sum: ", gamma_sum)

            # t+1 update
            for l in range(self.const['L']):
                self.muL[l] = (self.sum_muL(omega[l], self.x)) / (omega_sum[l])
                self.sigmaL2[l] = (2.0 * self.const['a'] + self.sum_sigmaL2(omega[l], self.x, self.muL[l])) \
                                            / (2.0 * self.const['b'] + omega_sum[l])
                self.pL[l] = (omega_sum[l]) / (gamma_sum)
                
            # find varphi(t+1) that satisfies Armijo condition and compute Q2(t+1) - Q2(t)
            self.computeArmijo(t, self.const['maxIter'])
            
            # check for parameter convergence (13) using phi and varphi
            if self.converged():
                self.convergence_count += 1
                if self.convergence_count == self.const['newton_max']:
                    print("converged at: ", t)
                    break
            else:
                self.convergence_count = 0
             
            self.end = time.time()
            print("time elapsed:", self.end - self.start)
            
        Q2_np = self.Q2.cpu().numpy()
        Q2_df = pd.DataFrame(Q2_np)
        Q2_df.to_csv(self.SAVE_DIR + '/deltaQ2.csv')
        
        # Save B, H, mu, sigma, pl, gamma
        params_directory = os.path.join(self.SAVE_DIR, 'params.txt')
        with open(params_directory, 'w') as outfile:
            np.savetxt(outfile, self.params.cpu().numpy(), fmt='%-8.4f')
            outfile.write('# pL\n')
            np.savetxt(outfile, self.pL.cpu().numpy(), fmt='%-8.4f')
            outfile.write('# sigmaL2\n')
            np.savetxt(outfile, self.sigmaL2.cpu().numpy(), fmt='%-8.4f')
            outfile.write('# muL\n')
            np.savetxt(outfile, self.muL.cpu().numpy(), fmt='%-8.4f')

        print("Computing 1e5 samples for LIS")
        # generate gamma using 10000 samples 
        ln_1 = 0
        for l in range(self.const['L']):
            ln_1 += ((self.pL[l]) / (torch.sqrt(2 * torch.pi * self.sigmaL2[l]))) * torch.exp(-(torch.pow(self.x - self.muL[l], 2)) / (2 * self.sigmaL2[l]))
        hs = torch.log(ln_1 / ln_0) 
        self.gamma = self.gibb_sampler_nocomp(100, 10000, hs)
        self.gamma /= 10000

        gamma_directory = os.path.join(self.SAVE_DIR, 'gamma.txt')
        with open(gamma_directory, 'w') as outfile:
            for data_slice in self.gamma.cpu().numpy():
                np.savetxt(outfile, data_slice, fmt='%-8.4f')
                outfile.write('# New z slice\n')

        print("Q2(max_iter) - Q(0): ", torch.sum(self.Q2))
        self.end = time.time()
        print("time elapsed:", self.end - self.start)

    def computeArmijo(self, epoch, max_epochs):
        lambda_m = 1.0
        diff = torch.zeros(5)
        count = 0
        satisfied = False
        # eq(11) satisfaction condition+

        delta = torch.matmul(self.I_inv, self.U)
        print("delta: ", delta)
        while count < 10:
            self.params = self.params_prev + lambda_m * delta
            print("est_params: ", self.params)
            # check for alternative criterion when delta_varphi is too small for armijo convergence
            # In practice, the Armijo condition (11) might not be satisfied
            # when the step length is too small
            diff = torch.abs(self.params - self.params_prev) / (torch.abs(self.params_prev) + self.const['eps1'])
            max_val = torch.amax(diff)
            if max_val < self.const['eps3']:
                print("alternative condition: ", max_val)
                self.params = self.params_prev
                break
            
            # use Gibbs sampler again to compute log_factor for delta_Q2 approximation
            log_factor = self.computeLogFactor()

            deltaQ2 = torch.matmul(self.params - self.params_prev, self.H_x) + log_factor
            self.Q2[epoch] = deltaQ2

            print("deltaQ2: [" + str(deltaQ2) + "]")
            if deltaQ2 > 0:
                satisfied = True
                break
            else:
                count += 1
                lambda_m /= 5
        return

    def computeLogFactor(self):

        theta_next, H_next_mean, H_next_iter, exp_sum_next, g = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], torch.zeros(self.x.shape))
        # compute log sum for varphi_next
        max_val = torch.amax(exp_sum_next)
        max_val = (max_val + self.log_sum_max_elem)/2.0

        diff_next = torch.sum(torch.exp(exp_sum_next - max_val))
        diff_curr = torch.sum(torch.exp(self.exp_sum - max_val))

        if not torch.isfinite(diff_next) or not torch.isfinite(diff_curr):
            return 0.0

        print("exp_sum_next - max_val: ", exp_sum_next - max_val)
        print("self.exp_sum - max_val: ", self.exp_sum - max_val)
        print("diff_next: ", diff_next)
        print("diff_curr: ", diff_curr)

        return torch.log(diff_next/diff_curr)

    def converged(self):
        # we have 8 parameters
        # B, H, pl[0], pl[1], mul[0], mul[1], sigmal[0], sigmal[1]
        diff_param = torch.abs(self.params - self.params_prev) / (torch.abs(self.params_prev) + self.const['eps1'])
        diff_muL = torch.abs(self.muL - self.muL_prev) / (torch.abs(self.muL_prev) + self.const['eps4'])
        diff_pL = torch.abs(self.pL - self.pL_prev) / (torch.abs(self.pL_prev) + self.const['eps4'])
        diff_sigmaL = torch.abs(self.sigmaL2 - self.sigmaL2_prev) / (torch.abs(self.sigmaL2_prev) + self.const['eps4'])
        diff = torch.cat([diff_param, diff_muL, diff_sigmaL, diff_pL], dim=0)
        print("Diff_param: ", diff)

        diff1 = torch.cat([diff_muL, diff_sigmaL, diff_pL], dim=0)

        _max1 = torch.amax(diff_param)
        _max2 = torch.amax(diff1)
        if _max1 < self.const['eps2'] and _max2 < self.const['eps5']:
            return True

        return False
        
    def gibb_sampler(self, burn_in, n_samples, hs):
        # initialize labels
        H_iter = torch.zeros((n_samples, 5))
        H_mean = torch.zeros(5)
        exp_sum = torch.zeros(n_samples)
        gamma = torch.zeros(self.x.shape)
        theta = self.init.clone()

        iteration = 0
        iter_burn = 0
        np_params = (self.params / self.H_reparam).cpu().numpy().astype('float64')
        np_hs = hs.cpu().numpy().astype('float64')
        while iteration < n_samples:
            theta = torch.from_numpy(run(Data.SIZE, np_params, theta.cpu().numpy().astype('float64'), self.theta_sum, self.t_ij, np_hs))
            if iter_burn < burn_in:
                iter_burn += 1
            else:
                
                theta_sum = torch.sum(theta)

                # do later
                H_iter[iteration][0] = -theta_sum
                H_mean[0] += H_iter[iteration][0]

                H_iter[iteration][1] = (-torch.sum(theta)*torch.sum(torch.ones((Data.SIZE,Data.SIZE,Data.SIZE))-theta))
                H_mean[1] += H_iter[iteration][1]

                h_i = torch.nonzero(theta).type(torch.DoubleTensor)
                h_j = torch.nonzero(torch.ones((Data.SIZE,Data.SIZE,Data.SIZE))-theta).type(torch.DoubleTensor)

                if h_i.shape[0] < 1 or h_j.shape[0] < 1:
                    print("...............H[iter] is 0")
                    tmp = torch.tensor([0.0])
                    H_iter[iteration][2] = tmp
                    H_mean[2] += H_iter[iteration][2]

                    H_iter[iteration][3] = tmp
                    H_mean[3] += H_iter[iteration][3]

                    H_iter[iteration][4] = tmp
                    H_mean[4] += H_iter[iteration][4]
                else:
                    tmp = torch.add(hi_dij(h_i, h_j), 1) # 1/(1+d_ij)
                    H_iter[iteration][2] = (-torch.sum(tmp**(-1)))
                    H_mean[2] += H_iter[iteration][2]

                    H_iter[iteration][3] = (-torch.sum(tmp**(-2)))
                    H_mean[3] += H_iter[iteration][3]

                    H_iter[iteration][4] = (-torch.sum(tmp**(-3)))
                    H_mean[4] += H_iter[iteration][4]

                H_mean /= self.H_reparam
                H_iter[iteration] /= self.H_reparam
                exp_sum[iteration] = -torch.sum(self.params*H_iter[iteration])
                # divde H(0.5) after calculating exp_sum
                
                gamma += theta

                iteration += 1
        return theta, H_mean, H_iter, exp_sum, gamma

    def gibb_sampler_nocomp(self, burn_in, n_samples, hs):
        # initialize labels
        gamma = torch.zeros(self.x.shape)
        theta = self.init.clone()

        iteration = 0
        iter_burn = 0
        np_params = (self.params / self.H_reparam).cpu().numpy().astype('float64')
        np_hs = hs.cpu().numpy().astype('float64')
        while iteration < n_samples:
            theta = torch.from_numpy(run(Data.SIZE, np_params, theta.cpu().numpy().astype('float64'), self.theta_sum, self.t_ij, np_hs))
            if iter_burn < burn_in:
                iter_burn += 1
            else:
                gamma += theta
                iteration += 1
        return gamma

    def compute_I_inverse(self, num_samples, H, H_mean):
        I = torch.matmul(torch.transpose(H-H_mean,0,-1), H-H_mean) / (num_samples - 1)
        print("I: ", I)
        determinant = torch.linalg.det(I)
        if determinant > 0:
            return torch.inverse(I)
        else:
            # identity matrix
            return torch.eye(5)

    def norm(self, x, mu, sigma2):
        return torch.exp(-1 * (x - mu) * (x - mu) / (2 * sigma2)) / (torch.sqrt(2 * torch.pi * sigma2))
    
    def sum_muL(self, omega, x):
        return torch.sum(omega * x)

    def sum_sigmaL2(self, omega, x, muL):       
        return torch.sum(omega * torch.pow(x - muL, 2))

    def reparameterization_factor(self):
        ones = torch.ones((Data.SIZE, Data.SIZE, Data.SIZE))
        h_i = torch.nonzero(ones).type(torch.DoubleTensor)
        h_j = torch.nonzero(ones).type(torch.DoubleTensor)
        dist = torch.add(hi_dij(h_i, h_j), 1)

        self.H_reparam[0] = -(Data.SIZE**3)*0.5
        self.H_reparam[1] = -(Data.SIZE**6)*0.25
        self.H_reparam[2] = -0.25*torch.sum(dist ** (-1))
        self.H_reparam[3] = -0.25*torch.sum(dist ** (-2))
        self.H_reparam[4] = -0.25*torch.sum(dist ** (-3))

        # uses cpu since it doesn't take long
    def p_lis(self, gamma_1, label):
        # LIS = P(theta = 0 | x)
        # gamma_1 = P(theta = 1 | x) = 1 - LIS
        dtype = [('index', float), ('value', float)]
        lis = np.zeros((Data.SIZE * Data.SIZE * Data.SIZE), dtype=dtype)
        for vx in range(Data.SIZE):
            for vy in range(Data.SIZE):
                for vk in range(Data.SIZE):
                    index = (vk * Data.SIZE * Data.SIZE) + (vy * Data.SIZE) + vx
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
            print(j, lis[j]['value'])
            if sum > (j+1)*self.const['fdr_control']:
                k = j
                break
        print("k", k)

        signal_lis = np.ones((Data.SIZE, Data.SIZE, Data.SIZE))
        for j in range(k):
            index = lis[j]['index']
            vk = index // (Data.SIZE*Data.SIZE)  # integer division
            index -= vk*Data.SIZE*Data.SIZE
            vy = index // Data.SIZE  # integer division
            vx = index % Data.SIZE
            vk = int(vk)
            vy = int(vy)
            vx = int(vx)
            signal_lis[vx][vy][vk] = 0  # reject these voxels, rest are 1

        # Compute FDR, FNR, ATP using LIS and Label
        # FDR -> (theta = 0) / num_rejected 
        # FNR -> (theta = 1) / num_not_rejected
        # ATP -> (theta = 1) that is rejected
        num_rejected = k
        num_not_rejected = (Data.SIZE*Data.SIZE*Data.SIZE) - k
        fdr = 0
        fnr = 0
        atp = 0
        for i in range(Data.SIZE):
            for j in range(Data.SIZE):
                for k in range(Data.SIZE):
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
        
        # Save final signal_file
        signal_directory = os.path.join(self.SAVE_DIR, 'lis.txt')
        with open(signal_directory, 'w') as outfile:
            outfile.write('fdr: ' + str(fdr) + '\n')
            outfile.write('fnr: ' + str(fnr) + '\n')
            outfile.write('atp: ' + str(atp) + '\n')
            for data_slice in signal_lis:
                np.savetxt(outfile, data_slice, fmt='%-8.4f')
                outfile.write('# New z slice\n')
        return

@vectorize([float64(float64, float64)], target ='cpu')
def hj_theta(t_ij, result):
    return t_ij * result

@numba.njit(cache = True, parallel=True)
def run(voxel_size, params, label, theta_sum, t_ij, hs):
    result = label
    for i in numba.prange(voxel_size):
        for j in numba.prange(voxel_size):
            for k in numba.prange(voxel_size):
                hj_t = hj_theta(t_ij[30*30*i + 30*j + k], result)
                hj_theta_sum = np.sum(hj_t)
                numerator = np.exp((-theta_sum[i][j][k] - params[0] + hs[i][j][k] + hj_theta_sum))

                p = numerator / (1 + numerator)
                if np.isnan(p): #because lim inf/1+inf ~ 1
                    result[i][j][k] = np.random.binomial(1, 1.0)
                else:
                    result[i][j][k] = np.random.binomial(1, p)
    return result

#https://nenadmarkus.com/p/all-pairs-euclidean/
def hi_dij(A, B):
    sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], B.shape[0])
    sqrB = torch.sum(torch.pow(B, 2), 1, keepdim=True).expand(B.shape[0], A.shape[0]).t()
    return torch.sqrt(sqrA - 2 * torch.mm(A, B.t()) + sqrB)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test_Statistics Generation')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    rng_seed = args.seed
    torch.manual_seed(rng_seed)
    np.random.seed(int(rng_seed))
    torch.set_default_dtype(torch.float64)
    numba.set_num_threads(int(numba.config.NUMBA_NUM_THREADS))

    fdr = Model2('../data/model2/' + str(rng_seed) +'/x.txt', rng_seed)
    fdr.gem()
    gamma_file = '../data/model2/' + str(rng_seed) +'/result' +'/gamma.txt'
    label_file = '../data/model2/' + str(rng_seed) +'/label/label.txt'
    fdr.p_lis(np.loadtxt(gamma_file).reshape((Data.SIZE,Data.SIZE,Data.SIZE)),
            np.loadtxt(label_file).reshape((Data.SIZE,Data.SIZE,Data.SIZE)))







