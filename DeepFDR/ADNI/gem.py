import numpy as np
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numba
from numba import cuda, float32, int32, guvectorize, vectorize, config, float64
import argparse
import ray 

class Model2:
    def __init__(self, x, savepath, num_cpus=1):
        #torch.set_default_dtype(torch.float64)
        self.num_cpus = num_cpus
        self.SAVE_DIR = os.path.join(savepath, 'result')
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        self.params = np.array([2e-3, 5e-7, 5e-4, 5e-3, 6e-3]).astype(np.float64)
        self.params_prev = np.array([2e-3, 5e-7, 5e-4, 5e-3, 6e-3]).astype(np.float64)

        self.pL = np.array([0.5, 0.5])
        self.muL = np.array([-2.5, 1.5])
        self.sigmaL2 = np.array([1.0, 1.0])
        self.pL_prev = np.zeros(2)
        self.muL_prev = np.zeros(2)
        self.sigmaL2_prev = np.zeros(2)

        self.const = {'a': 1,  # for penalized likelihood for L >= 2
                      'b': 2,
                      'delta': 1e-3,
                      'maxIter': 1,
                      'eps1': 1e-4,
                      'eps2': 0.05,
                      'eps3': 0.02,
                      'eps4': 1e-3,
                      'eps5': 1e-2,
                      'alpha': 1e-3,
                      'burn_in': 80, # 80
                      'num_samples': 100, # 100
                      'L': 2,
                      'newton_max': 3,
                      'fdr_control': 0.1,
                      }

        self.x = x
        self.mapping = np.load(os.path.join(args.savepath, '1d_roi_to_3d_mapping.npy'))
        self.size = self.x.shape[0]
        self.gamma = np.zeros(self.size)  # P(theta | x)
        self.init = np.zeros(self.size)  # initial theta value for gibb's sampling
        self.H_x = np.zeros(5).astype(np.float64)
        self.H_mean = np.zeros(5).astype(np.float64)
        self.log_sum_max_elem = np.zeros(1).astype(np.float64)
        self.exp_sum = np.zeros(self.const['num_samples']).astype(np.float64)
        self.U = np.zeros(5).astype(np.float64)
        self.I_inv = np.zeros((5, 5)).astype(np.float64)
        self.H_iter = np.zeros((self.const['num_samples'], 5)).astype(np.float64)
        self.H_x_iter = np.zeros((self.const['num_samples'], 5)).astype(np.float64)
        self.convergence_count = 0
        init_prob = np.full((self.size), 0.5)
        self.init = np.random.binomial(1, init_prob).astype('float64')
        self.theta = self.init.copy()

        self.H_reparam = np.zeros(5).astype('float64')

        self.reparameterization_factor()

        self.params = self.params * self.H_reparam
        self.params_prev = self.params_prev * self.H_reparam

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
        const_0 = 1 / np.sqrt(np.array([2 * np.pi * sigma0_sq]))
        ln_0 = const_0 * np.exp(-(np.power(self.x - mu_0, 2)) / (2 * sigma0_sq))
        ln_0 = ln_0.astype(np.float64)

        for t in range(self.const['maxIter']):
            print("")
            print("GEM iter: ", t)
            print("params:  " + str(self.params))
            print("params_prev: " + str(self.params_prev))
            print("pL:  " + str(self.pL) + " pL_prev: " + str(self.pL_prev))
            print("muL:  " + str(self.muL) + " muL_prev: " + str(self.muL_prev))
            print("sigmaL2:  " + str(self.sigmaL2) + " sigmaL2_prev: " + str(self.sigmaL2_prev))

            # varphi: (10), (11)
            self.params_prev = self.params.copy()
            self.muL_prev = self.muL.copy()
            self.sigmaL2_prev = self.sigmaL2.copy()
            self.pL_prev = self.pL.copy()

            # calculate hs(t)
            ln_1 = np.float64(0)
            for l in range(self.const['L']):
                ln_1 += ((self.pL[l]) / (np.sqrt(2 * np.pi * self.sigmaL2[l]))) * np.exp(-(np.power(self.x - self.muL[l], 2)) / (2 * self.sigmaL2[l]))

            # ----------------------------------------CHECKPOINT MARKER-------------------------------------------------------#
            # Gibb's sampler to generate theta and theta_x
            self.theta, self.H_mean, self.H_iter, self.exp_sum, g = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], np.zeros(self.size))
            hs = np.log(ln_1 / ln_0)

            if np.isnan(hs).any():
                print('NAN inside hs')

            theta_x, self.H_x, self.H_x_iter, exp_x_sum, self.gamma = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], hs)

            # Monte carlo averages
            self.H_mean /= self.const['num_samples']
            self.H_x /= self.const['num_samples']
            self.gamma /= self.const['num_samples']
            gamma_sum = np.sum(self.gamma)


            print("H_mean: ", self.H_mean)
            print("H_x: ", self.H_x)
            print("H_iter: ", self.H_iter[0:5])
            print("H_meaaaan: ", np.sum(self.H_iter, axis=0))
            # compute log factor for current varphi
            # Because np.exp can overflow with large value of exp_sum, a modification is made
            # where we subtract by max_val. Same is done for log_sum_next
            self.log_sum_max_elem = np.amax(self.exp_sum)
            if np.isnan(self.exp_sum).any():
                print('NAN inside self.exp_sum')
            # compute U, I
            # U --> 5x1 matrix, I --> 5x5 matrix, trans(U)*inv(I)*U = 1x1 matrix
            self.U = self.H_x - self.H_mean
            print("U: ", self.U)

            self.I_inv = self.compute_I_inverse(self.const['num_samples'], self.H_iter, self.H_mean)
            print('I_inv --> ', self.I_inv)


            # phi: (6), (7), (8)
            f_x = np.zeros(self.size)
            omega_sum = np.zeros(self.const['L'])
            omega = np.empty((2, self.size))

            for l in range(self.const['L']):
                f_x += self.pL[l] * self.norm(self.x, self.muL[l], self.sigmaL2[l])
            for l in range(self.const['L']):
                omega[l] = self.gamma * self.pL[l] / f_x * self.norm(self.x, self.muL[l], self.sigmaL2[l])
                omega_sum[l] = np.sum(omega[l])

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


        # Save B, H, mu, sigma, pl, gamma
        params_directory = os.path.join(self.SAVE_DIR, 'params.txt')
        with open(params_directory, 'w') as outfile:
            np.savetxt(outfile, self.params, fmt='%-8.4f')
            outfile.write('# pL\n')
            np.savetxt(outfile, self.pL, fmt='%-8.4f')
            outfile.write('# sigmaL2\n')
            np.savetxt(outfile, self.sigmaL2, fmt='%-8.4f')
            outfile.write('# muL\n')
            np.savetxt(outfile, self.muL, fmt='%-8.4f')


        self.end = time.time()
        print("time elapsed:", self.end - self.start)

    def computeArmijo(self, epoch, max_epochs):
        lambda_m = 1.0
        diff = np.zeros(5)
        count = 0
        satisfied = False
        # eq(11) satisfaction condition+

        delta = np.matmul(self.I_inv, self.U)
        print("delta: ", delta)
        while count < 10:
            self.params = self.params_prev + lambda_m * delta
            print("est_params: ", self.params)
            # check for alternative criterion when delta_varphi is too small for armijo convergence
            # In practice, the Armijo condition (11) might not be satisfied
            # when the step length is too small
            diff = np.abs(self.params - self.params_prev) / (np.abs(self.params_prev) + self.const['eps1'])
            max_val = np.amax(diff)
            if max_val < self.const['eps3']:
                print("alternative condition: ", max_val)
                self.params = self.params_prev
                break

            # use Gibbs sampler again to compute log_factor for delta_Q2 approximation
            log_factor = self.computeLogFactor()

            deltaQ2 = np.matmul(self.params - self.params_prev, self.H_x) + log_factor

            print("deltaQ2: [" + str(deltaQ2) + "]")
            if deltaQ2 > 0:
                satisfied = True
                break
            else:
                count += 1
                lambda_m /= 5
        return

    def computeLogFactor(self):

        theta_next, H_next_mean, H_next_iter, exp_sum_next, g = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], np.zeros(self.size))
        # compute log sum for varphi_next
        max_val = np.amax(exp_sum_next)
        max_val = (max_val + self.log_sum_max_elem)/2.0

        diff_next = np.sum(np.exp(exp_sum_next - max_val))
        diff_curr = np.sum(np.exp(self.exp_sum - max_val))

        if not np.isfinite(diff_next) or not np.isfinite(diff_curr):
            return 0.0

        return np.log(diff_next/diff_curr)

    def converged(self):
        # we have 8 parameters
        # B, H, pl[0], pl[1], mul[0], mul[1], sigmal[0], sigmal[1]
        diff_param = np.abs(self.params - self.params_prev) / (np.abs(self.params_prev) + self.const['eps1'])
        diff_muL = np.abs(self.muL - self.muL_prev) / (np.abs(self.muL_prev) + self.const['eps4'])
        diff_pL = np.abs(self.pL - self.pL_prev) / (np.abs(self.pL_prev) + self.const['eps4'])
        diff_sigmaL = np.abs(self.sigmaL2 - self.sigmaL2_prev) / (np.abs(self.sigmaL2_prev) + self.const['eps4'])
        diff = np.hstack((diff_param, diff_muL, diff_sigmaL, diff_pL))
        diff1 = np.hstack((diff_muL, diff_sigmaL, diff_pL))

        _max1 = np.amax(diff_param)
        _max2 = np.amax(diff1)
        if _max1 < self.const['eps2'] and _max2 < self.const['eps5']:
            return True

        return False

    def gibb_sampler(self, burn_in, n_samples, hs):
        iter_burn = 0
        H_iter = exp_sum = None
        H_mean = np.zeros(5)
        gamma = np.zeros(self.size)
        #theta = self.init.copy()
        params = self.params / self.H_reparam

        while iter_burn < burn_in:
            self.theta = run(self.size, params, self.theta, dist, hs)
            iter_burn += 1

        if n_samples > 0:
            # make sure dist_id is in shared_memory using ray.put
            # modify code to allow flexible num_cpus and n_samples
            results = [sample_ray.remote(self.size, params, self.theta, dist_id, self.mapping, self.H_reparam, n_samples//self.num_cpus, hs) for i in range(self.num_cpus)]

            done = False
            while not done:
                ready, not_ready = ray.wait(results)
                get = ray.get(ready)
                if H_iter is None:
                    H_iter = np.asarray(get[0][0])
                else:
                    H_iter = np.vstack((H_iter, np.asarray(get[0][0])))

                H_mean = H_mean + np.asarray(get[0][1])

                if exp_sum is None:
                    exp_sum = np.asarray(get[0][0])
                else:
                    exp_sum = np.vstack((exp_sum, np.asarray(get[0][0])))

                gamma = gamma + np.asarray(get[0][3])

                results = not_ready
                if not results: 
                    done = True

        #print(self.theta, H_mean, H_iter, exp_sum, gamma)
        return self.theta, H_mean, H_iter, exp_sum, gamma


    def gibb_sampler_nocomp(self, burn_in, n_samples, params, pL, sigmaL2, muL, x):
        mu_0 = 0
        sigma0_sq = 1
        const_0 = 1 / np.sqrt(np.array([2 * np.pi * sigma0_sq]))
        ln_0 = const_0 * np.exp(-(np.power(x - mu_0, 2)) / (2 * sigma0_sq))

        # calculate hs(t)
        ln_1 = 0
        for l in range(self.const['L']):
            ln_1 += ((pL[l]) / (np.sqrt(2 * np.pi * sigmaL2[l]))) * np.exp(-(np.power(x - muL[l], 2)) / (2 * sigmaL2[l]))
        # ----------------------------------------CHECKPOINT MARKER-------------------------------------------------------#
        # Gibb's sampler to generate theta and theta_x
        hs = np.log(ln_1 / ln_0)

        # initialize labels
        gamma = np.zeros(self.size)
        theta = self.init.copy()
        iter_burn = 0
        params /= self.H_reparam
        while iter_burn < burn_in:
            theta = run(self.size, params, theta, dist, hs)
            iter_burn += 1

        if n_samples > 0:
            # make sure dist_id is in shared_memory using ray.put
            # modify code to allow flexible num_cpus and n_samples
            results = [sample_ray_nocomp.remote(self.size, params, theta, dist_id, n_samples//self.num_cpus, hs) for i in range(self.num_cpus)]

            done = False
            while not done:
                ready, not_ready = ray.wait(results)
                get = ray.get(ready)
                gamma = gamma + np.asarray(get)
                results = not_ready
                if not results: 
                    done = True

        gamma /= n_samples
        return gamma

    def compute_I_inverse(self, num_samples, H, H_mean):
        I = np.matmul(np.transpose(H-H_mean), H-H_mean) / (num_samples - 1)
        determinant = np.linalg.det(I)
        if determinant > 0:
            return np.linalg.inv(I)
        else:
            # identity matrix
            return np.eye(5)

    def norm(self, x, mu, sigma2):
        return np.exp(-1 * (x - mu) * (x - mu) / (2 * sigma2)) / (np.sqrt(2 * np.pi * sigma2))

    def sum_muL(self, omega, x):
        return np.sum(omega * x)

    def sum_sigmaL2(self, omega, x, muL):
        return np.sum(omega * np.power(x - muL, 2))

    def reparameterization_factor(self):
        dist = hi_dij(self.mapping, self.mapping) + 1 # H(0.5), so all voxels are used
        self.H_reparam[0] = -(self.size)*0.5
        self.H_reparam[1] = -(self.size**2)*0.25
        self.H_reparam[2] = -0.25*np.sum(dist ** (-1))
        self.H_reparam[3] = -0.25*np.sum(dist ** (-2))
        self.H_reparam[4] = -0.25*np.sum(dist ** (-3))


    def p_lis(self, gamma_1, label=None):
        '''
        Rejection of null hypothesis are shown as 1, consistent with online BH, Q-value, smoothFDR methods.
        # LIS = P(theta = 0 | x)
        # gamma_1 = P(theta = 1 | x) = 1 - LIS
        '''
        gamma_1 = gamma_1.squeeze()
        print(gamma_1.shape, label.shape)
        dtype = [('index', int), ('value', float)]
        lis = np.zeros(self.size, dtype=dtype)
        for i in range(self.size):
            lis[i]['index'] = i
            lis[i]['value'] = 1 - gamma_1[i]
        # sort using lis values
        lis = np.sort(lis, order='value')
        # Data driven LIS-based FDR procedure
        sum = 0
        k = 0
        for j in range(len(lis)):
            sum += lis[j]['value']
            if sum > (j+1)*self.const['fdr_control']:
                k = j
                break

        signal_lis = np.zeros(self.size)
        for j in range(k):
            index = lis[j]['index']
            signal_lis[index] = 1  

        np.save(os.path.join(self.SAVE_DIR, 'lis.npy'), signal_lis)

        print(self.size)
        print(k)

        if label is not None:
            # Compute FDR, FNR, ATP using LIS and Label
            # FDR -> (theta = 0) / num_rejected
            # FNR -> (theta = 1) / num_not_rejected
            # ATP -> (theta = 1) that is rejected
            num_rejected = k
            num_not_rejected = self.size - k
            fdr = 0
            fnr = 0
            atp = 0
            for i in range(self.size):
                if signal_lis[i] == 1: # rejected
                    if label[i] == 0:
                        fdr += 1
                    elif label[i] == 1:
                        atp += 1
                elif signal_lis[i] == 0: # not rejected
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
            signal_directory = os.path.join(self.SAVE_DIR, 'stats.txt')
            with open(signal_directory, 'w') as outfile:
                outfile.write('Model2: \n')
                outfile.write('fdr: ' + str(fdr) + '\n')
                outfile.write('fnr: ' + str(fnr) + '\n')
                outfile.write('atp: ' + str(atp) + '\n')


@vectorize([float64(float64, float64, float64, float64, float64)], target ='cpu')
def theta_ij(dist, B_1, B_1d, B_2d, B_3d):
    #return (B_1 + B_3d * dist*dist*dist + B_3r * rho*rho*rho + B_2d * dist*dist + B_2r * rho*rho + B_1d * dist + B_1r * rho)
    return B_1 + B_3d*dist*dist*dist + B_2d*dist*dist + B_1d*dist

@vectorize([float64(float64, float64)], target ='cpu')
def hj_theta(t_ij, result):
    return t_ij * result

@numba.njit(cache = True, parallel=True)
def run(voxel_size, params, label, dist, hs):
    result = label
    for i in numba.prange(voxel_size):
        t_ij = theta_ij(dist[i], params[1], params[2], params[3], params[4])
        theta_sum = np.sum(t_ij)

        hj_t = hj_theta(t_ij, result)
        hj_theta_sum = np.sum(hj_t)

        numerator = np.exp(-theta_sum - params[0] + hs[i] + hj_theta_sum)

        p = numerator / (1 + numerator)
        if np.isnan(p): #because lim inf/1+inf ~ 1
            result[i] = np.random.binomial(1, 1.0)
        else:
            result[i] = np.random.binomial(1, p)
    return result

@numba.njit(cache = True, parallel=True)
def sample_using_workers(voxel_size, params, label, dist, epochs, hs):
    results = np.empty((epochs, voxel_size))
    for e in range(epochs):
        result = np.copy(label)
        for i in numba.prange(voxel_size):
            t_ij = theta_ij(dist[i], params[1], params[2], params[3], params[4])
            theta_sum = np.sum(t_ij)

            hj_t = hj_theta(t_ij, result)
            hj_theta_sum = np.sum(hj_t)

            numerator = np.exp(-theta_sum - params[0] + hs[i] + hj_theta_sum)

            p = numerator / (1 + numerator)
            if np.isnan(p): #because lim inf/1+inf ~ 1
                result[i] = np.random.binomial(1, 1.0)
            else:
                result[i] = np.random.binomial(1, p)
        results[e] = result
    return results

@ray.remote
def sample_ray_nocomp(voxel_size, params, label, dist, epochs, hs):
    theta = sample_using_workers(voxel_size, params, label, dist, epochs, hs)
    gamma = np.sum(theta, axis=0) # (epochs,1)
    return gamma

@ray.remote
def sample_ray(voxel_size, params, label, dist, mapping, H_reparam, epochs, hs):
    theta = sample_using_workers(voxel_size, params, label, dist, epochs, hs)

    H_iter = np.zeros((epochs, 5))
    H_mean = np.zeros(5)
    exp_sum = np.zeros(epochs)
    gamma = np.zeros(voxel_size)

    # theta has shape (epochs, voxel_size)
    gamma = np.sum(theta, axis=0) # (voxel_size, 1)
    theta_sum = -np.sum(theta, axis=1)  # (epochs,1)

    H_iter[:,0] = theta_sum

    opp_theta = 1-theta
    H_iter[:,1] = theta_sum*np.sum(opp_theta, axis=1)

    for iteration in range(epochs):
        h_i = np.transpose(np.nonzero(theta[iteration]))
        h_j = np.transpose(np.nonzero(opp_theta[iteration]))
        h_i = np.squeeze(mapping[h_i], axis=1)
        h_j = np.squeeze(mapping[h_j], axis=1)
        if h_i.shape[0] < 1 or h_j.shape[0] < 1:
            tmp = np.array([0.0])
            H_iter[iteration][2] = tmp

            H_iter[iteration][3] = tmp

            H_iter[iteration][4] = tmp
        else:
            tmp = hi_dij(h_i, h_j) + 1 # 1/(1+d_ij)
            H_iter[iteration][2] = (-np.sum(tmp**(-1)))

            H_iter[iteration][3] = (-np.sum(tmp**(-2)))

            H_iter[iteration][4] = (-np.sum(tmp**(-3)))

        H_iter[iteration] /= H_reparam
        H_mean += H_iter[iteration]
        exp_sum[iteration] = -np.sum(params*H_iter[iteration])

        iteration += 1

    return (H_iter, H_mean, exp_sum, gamma)
 

#https://nenadmarkus.com/p/all-pairs-euclidean/
def hi_dij(A, B):
    sqrA = np.broadcast_to(np.sum(np.power(A, 2), 1).reshape(A.shape[0], 1), (A.shape[0], B.shape[0]))
    sqrB = np.broadcast_to(np.sum(np.power(B, 2), 1).reshape(B.shape[0], 1), (B.shape[0], A.shape[0])).transpose()
    return np.sqrt(sqrA - 2*np.matmul(A, B.transpose()) + sqrB)

def parse_params(params_file):
    p = np.zeros(2)
    mu = np.zeros(2)
    sigma = np.zeros(2)

    with open(params_file, 'r') as file:
        lines = file.readlines()
        params = np.array([float64(i.strip()) for i in lines[0:5]])
        pL = np.array([float64(i.strip()) for i in lines[6:8]])
        sigmaL2 = np.array([float64(i.strip()) for i in lines[9:11]])
        muL = np.array([float64(i.strip()) for i in lines[12:14]])

    return params, pL, sigmaL2, muL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test_Statistics Generation')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--mode', type=str, help='Supported modes: gem_gen, label_gen', required=True)
    parser.add_argument('--x_path', type=str)
    parser.add_argument('--savepath', type=str)
    parser.add_argument('--simulation_path', type=str)
    parser.add_argument('--limit', default=600, type=int)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--num_cpus', type=int)
    args = parser.parse_args()

    num_files_per_section = 600
    rng_seed = args.seed
    limit = args.limit
    mode = args.mode

    np.random.seed(rng_seed)
    numba.set_num_threads(int(numba.config.NUMBA_NUM_THREADS))

    num_cpus = args.num_cpus

    ray.init(num_cpus=num_cpus)

    if mode == "label_gen": 
        npzfile = np.load(os.path.join(args.x_path, 'data.npz'))
        fdr = Model2(npzfile['arr_0'][0], args.savepath)
        label_npzarray = np.empty((npzfile['arr_0'].shape[0], 30, 30, 30))
        params_directory = os.path.join(args.savepath, 'result/params.txt')
        params, pL, sigmaL2, muL = parse_params(params_directory)

        for i in range(limit):        
            x = torch.from_numpy(npzfile['arr_0'][num_files_per_section*rng_seed + i])
            # burn_in = 80, n_samples = 2000 for simulation
            gamma = fdr.gibb_sampler_nocomp(burn_in=100, n_samples=2000, params=params, pL=pL, sigmaL2=sigmaL2, muL=muL, x=x)
            label_npzarray[num_files_per_section*rng_seed + i] = gamma.cpu().numpy()            

        train_label_directory = os.path.join(args.x_path, 'label.npz') 
        if not os.path.exists(train_label_directory):
            np.savez(train_label_directory, label_npzarray)
        else:
            orig = np.load(train_label_directory)
            orig_data = orig['arr_0']
            orig_data[num_files_per_section*rng_seed:num_files_per_section*rng_seed + limit] = label_npzarray[num_files_per_section*rng_seed:num_files_per_section*rng_seed + limit]
            np.savez(train_label_directory, orig_data)
            
    elif mode == "single_label":
        x = np.load(os.path.join(args.x_path, 'x_1d.npy'))
        model = Model2(x, args.savepath, num_cpus)

        dist = np.memmap(os.path.join(args.savepath, 'distance.npy'), dtype='float16', mode='r', shape=(model.size, model.size)).astype(np.float32) # -----------------------------ADD RAY
        global dist_id
        dist_id = ray.put(dist)

        model.gem()

        params_directory = os.path.join(args.savepath, 'result/params.txt')
        params, pL, sigmaL2, muL = parse_params(params_directory)

        start = time.time()
        gamma = model.gibb_sampler_nocomp(burn_in=100, n_samples=2000, params=params, pL=pL, sigmaL2=sigmaL2, muL=muL, x=x)
        np.save(os.path.join(args.savepath, 'result/gamma.npy'), gamma)
        model.p_lis(gamma)
        end = time.time()
        print('time elapsed for 2000 gibbs samples: ', end-time)
    ray.shutdown()
