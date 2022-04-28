import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numba
from numba import cuda, float32, int32, guvectorize, vectorize, config, float64
import argparse
import ray 
import rpy2
import rpy2.situation
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import scipy as sp
import ctypes
import nibabel as nib

class Model2:
    def __init__(self, x, savepath, num_cpus=1):
        #torch.set_default_dtype(torch.float64)
        self.num_cpus = num_cpus
        self.SAVE_DIR = os.path.join(savepath, 'result_gem_corr_gem')
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        self.params = np.array([2e-3, 5e-7, 5e-4, 5e-3, 6e-3, 6e-6, 8e-6, 6e-6, 6e-6, 8e-6, 6e-6]).astype('float64')
        self.params_prev = self.params

        self.pL = np.array([0.2, 0.2,0.2,0.2,0.2])
        self.muL = np.array([-2.0, 2.0,-4,4,0])
        self.sigmaL2 = np.array([0.25, 0.25,0.25, 0.25,0.25])
        self.pL_prev = np.zeros(5)
        self.muL_prev = np.zeros(5)
        self.sigmaL2_prev = np.zeros(5)
        self.lamda = 0.08
        self.const = {'a': 1,  # for penalized likelihood for L >= 2
                      'b': 2,
                      'delta': 1e-3,
                      'maxIter': 5,
                      'eps1': 1e-4,
                      'eps2': 0.05,
                      'eps3': 0.02,
                      'eps4': 1e-3,
                      'eps5': 1e-2,
                      'alpha': 1e-3,
                      'burn_in': 5, # 80
                      'num_samples': 5, # 100
                      'L': 5,
                      'newton_max': 3,
                      'fdr_control': 1e-3,
                      }

        self.x = x
        self.mapping = np.load(os.path.join(args.savepath, '1d_roi_to_3d_mapping.npy'))
        self.ln_1_prev = np.zeros(self.x.shape) # initial nonnull distribution
        self.ln_1 = np.zeros(self.x.shape) 
        self.size = self.x.shape[0]
        self.gamma = np.zeros(self.size)  # P(theta | x)
        self.ks = initial_ks(self.size,0.5,self.gamma)
        self.sum_replicate_count = np.sum(self.ks)
        self.init = np.zeros(self.size)  # initial theta value for gibb's sampling
        self.H_x = np.zeros(11).astype(np.float64)
        self.H_mean = np.zeros(11).astype(np.float64)
        self.log_sum_max_elem = np.zeros(1).astype(np.float64)
        self.exp_sum = np.zeros(self.const['num_samples']).astype(np.float64)
        self.U = np.zeros(11).astype(np.float64)
        self.I_inv = np.zeros((11, 11)).astype(np.float64)
        self.H_iter = np.zeros((self.const['num_samples'], 11)).astype(np.float64)
        self.H_x_iter = np.zeros((self.const['num_samples'], 11)).astype(np.float64)
        self.convergence_count = 0
        #torch.pi = torch.acos(torch.zeros(1)).item() * 2
        init_prob = np.full((self.size), 0.5)
        self.init = np.random.binomial(1, init_prob).astype('float64')
        self.theta = self.init.copy()

        self.H_reparam = np.zeros(11).astype('float64')

        self.reparameterization_factor()

        self.params = self.params * self.H_reparam
        self.params_prev = self.params_prev * self.H_reparam

   


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


        # calculate hs(t)
        ln_1 = np.float64(0)
        for l in range(self.const['L']):
            ln_1 += ((self.pL[l]) / (np.sqrt(2 * np.pi * self.sigmaL2[l]))) * np.exp(-(np.power(self.x - self.muL[l], 2)) / (2 * self.sigmaL2[l]))

        for t in range(self.const['maxIter']):
            print("")
            print("GEM iter: ", t)
            print("params:  " + str(self.params))
            print("params_prev: " + str(self.params_prev))

    
            # varphi: (10), (11)
            self.params_prev = self.params.copy()
            self.ln_1_prev = self.ln_1.copy()
          
            
            #calculate ln1 using R package "gss"
            #firstly determine lambda by cross validation
            lamda = self.lamda
            # m = 100m*(number of Gibbs samplers)
            m = 100 * self.const['num_samples']
    
            # lamda* = 2 * m * lamda / N
            lamda_star = 2 * m * lamda / self.sum_replicate_count
            
            flat_x = self.x.astype('float64')

            


            rpy2.robjects.numpy2ri.deactivate()
            np.savetxt(self.SAVE_DIR + "/dx.txt",flat_x,fmt='%1.3f')
            rreadtable = robjects.r['read.table']
            xr = rreadtable(self.SAVE_DIR + "/dx.txt")
            xr = xr.rx2(1)
            rpy2.robjects.numpy2ri.activate()
            robjects.globalenv["xr"] = xr

            r = robjects.r
            r.source('./sspdsty_shu.R')
            r.source('./ssden_shu.R')
            r.source('./nonnull_pdf.R')
            
            if t ==0:
                #ks = self.ks.cpu().numpy().flatten().astype('float64')
                #robjects.globalenv["ks"] = self.ks.cpu().numpy().flatten().astype('float64')
                self.ln_1 = ln_1
            else:
                #save the normal form of lambda_star and lambda for every iteration given lambda tilda
                robjects.globalenv["ks"] = ks
                robjects.globalenv["lamda_star"] = lamda_star
                pdf = r('nonnull_pdf')(x= xr, ks = ks, lambda_star =lamda_star)
                pdf = np.reshape(pdf,self.size, order='C')
                #pdf = torch.from_numpy(pdf)
                self.ln_1 = pdf

            # ----------------------------------------CHECKPOINT MARKER-------------------------------------------------------#
            # Gibb's sampler to generate theta and theta_x
            self.theta, self.H_mean, self.H_iter, self.exp_sum, g = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], np.zeros(self.size))
            hs = np.log(ln_1 / ln_0)

            if np.isnan(hs).any():
                print('NAN inside hs')

            theta_x, self.H_x, self.H_x_iter, exp_x_sum, self.gamma = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], hs)

            ks = get_ks(self.size,self.gamma)
            self.sum_replicate_count = np.sum(ks) #sum(ks), used as N
            #ks = ks.cpu().numpy().astype('float64')
            #self.sum_replicate_count = self.sum_replicate_count.cpu().numpy().astype('float64')

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

   


       
        params_directory = os.path.join(self.SAVE_DIR, 'params.txt')
        with open(params_directory, 'w') as outfile:
            np.savetxt(outfile, self.params, fmt='%-8.4f')
    

        np.save(os.path.join(self.SAVE_DIR, 'ks.npy'),ks)

    def computeArmijo(self, epoch, max_epochs):
        lambda_m = 1
        diff = np.zeros(11)
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

        diff_param = np.abs(self.params - self.params_prev) / (np.abs(self.params_prev) + self.const['eps1'])

        diff_nonmull_prob = np.abs(self.ln_1 - self.ln_1_prev) / (np.abs(self.ln_1_prev) + self.const['eps4'])

        print("Diff_param: ", diff_param)
        
        #diff1 = torch.cat([diff_muL, diff_sigmaL, diff_pL], dim=0)

        _max1 = np.amax(diff_param)
        _max2 = np.amax(diff_nonmull_prob)
        print("max_diff_nonmull_prob: ", _max2)
        if _max1 < self.const['eps2'] and _max2 < self.const['eps4']:
            return True

        return False

    def gibb_sampler(self, burn_in, n_samples, hs):
        iter_burn = 0
        H_iter = exp_sum = None
        H_mean = np.zeros(11)
        gamma = np.zeros(self.size)
        #theta = self.init.copy()
        params = self.params / self.H_reparam


        while iter_burn < burn_in:
            self.theta = run(self.size, params, self.theta, dist, hs, corr_d, corr_c)
            iter_burn += 1

        if n_samples > 0:
            # make sure dist_id is in shared_memory using ray.put
            # modify code to allow flexible num_cpus and n_samples
            results = [sample_ray.remote(self.size, params, self.theta, dist_id, self.mapping, self.H_reparam, n_samples//self.num_cpus, hs, corr_d_id, corr_c_id) for i in range(self.num_cpus)]

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


    def gibb_sampler_nocomp(self, burn_in, n_samples, params, ks, x):
        mu_0 = 0
        sigma0_sq = 1
        const_0 = 1 / np.sqrt(np.array([2 * np.pi * sigma0_sq]))
        ln_0 = const_0 * np.exp(-(np.power(x - mu_0, 2)) / (2 * sigma0_sq))

        m = 100 * self.const['num_samples']

        #lamda* = 2 * m * lamda / N
        lamda = self.lamda
        sum_replicate_count = np.sum(ks) #sum(ks), used as N
        lamda_star = 2 * m * lamda / sum_replicate_count
        rpy2.robjects.numpy2ri.activate()

        r = robjects.r
        r.source('./nonnull_pdf.R')

        
        robjects.globalenv["ks"] = ks
        robjects.globalenv["lamda_star"] = lamda_star


        flat_x = self.x.astype('float64')

            


        rpy2.robjects.numpy2ri.deactivate()
        np.savetxt(self.SAVE_DIR + "/dx.txt",flat_x,fmt='%1.3f')
        rreadtable = robjects.r['read.table']
        xr = rreadtable(self.SAVE_DIR + "/dx.txt")
        xr = xr.rx2(1)
        rpy2.robjects.numpy2ri.activate()
        robjects.globalenv["xr"] = xr

        pdf = r('nonnull_pdf')(x= xr, ks = ks, lambda_star =lamda_star)
        pdf = np.reshape(pdf,self.size, order='C')
        #pdf = torch.from_numpy(pdf)
        ln_1 = pdf


        hs = np.log(ln_1 / ln_0) 
        # initialize labels
        gamma = np.zeros(self.size)
        theta = self.init.copy()
        iter_burn = 0
        params /= self.H_reparam
        while iter_burn < burn_in:
            theta = run(self.size, params, theta, dist, hs,corr_d, corr_c)
            iter_burn += 1

        if n_samples > 0:
            # make sure dist_id is in shared_memory using ray.put
            # modify code to allow flexible num_cpus and n_samples
            results = [sample_ray_nocomp.remote(self.size, params, theta, dist_id, n_samples//self.num_cpus, hs, corr_d_id, corr_c_id) for i in range(self.num_cpus)]

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
            return np.eye(11)

    def norm(self, x, mu, sigma2):
        return np.exp(-1 * (x - mu) * (x - mu) / (2 * sigma2)) / (np.sqrt(2 * np.pi * sigma2))

    def sum_muL(self, omega, x):
        return np.sum(omega * x)

    def sum_sigmaL2(self, omega, x, muL):
        return np.sum(omega * np.power(x - muL, 2))

    def reparameterization_factor(self):
        dist = hi_dij(self.mapping, self.mapping) + 1 # H(0.5), so all voxels are used
        corr_c = np.memmap('/scratch/qj2022/ADNI_PET/corr_precompute_CN/correlation_CN_all.npy', dtype='float16', mode='r', shape=(self.size, self.size)).astype(np.float32)
        corr_d = np.memmap('/scratch/qj2022/ADNI_PET/corr_precompute_AD/correlation_AD_all.npy', dtype='float16', mode='r', shape=(self.size, self.size)).astype(np.float32) # -----------------------------ADD RAY
        
        self.H_reparam[0] = -(self.size)*0.5
        self.H_reparam[1] = -(self.size**2)*0.25
        self.H_reparam[2] = -0.25*np.sum(dist ** (-1))
        self.H_reparam[3] = -0.25*np.sum(dist ** (-2))
        self.H_reparam[4] = -0.25*np.sum(dist ** (-3))
        self.H_reparam[5] = -0.25*np.sum(corr_d ** (3)) #disease 3
        self.H_reparam[6] = -0.25*np.sum(corr_d ** (2)) #disease 2
        self.H_reparam[7] = -0.25*np.sum(corr_d ** (1)) #disease 1
        self.H_reparam[8] = -0.25*np.sum(corr_c ** (3)) #control 3
        self.H_reparam[9] = -0.25*np.sum(corr_c ** (2)) #control 2
        self.H_reparam[10] = -0.25*np.sum(corr_c ** (1)) #control 1


    def p_lis(self, gamma_1, label=None):
        '''
        Rejection of null hypothesis are shown as 1, consistent with online BH, Q-value, smoothFDR methods.
        # LIS = P(theta = 0 | x)
        # gamma_1 = P(theta = 1 | x) = 1 - LIS
        '''
        gamma_1 = gamma_1.squeeze()
        print(gamma_1.shape)
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
        print("number of signals")
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


@vectorize([float64(float64, float64, float64, float64, float64, float64, float64, float64,float64,float64, float64,float64,float64)], target ='cpu')
def theta_ij(dist, rho_d, rho_c,B_1, B_1d, B_2d, B_3d, B_11r, B_12r, B_13r,B_21r, B_22r, B_23r):
    dist_component = B_3d / ((1.0+dist)*(1.0+dist)*(1.0+dist)) + B_2d / ((1.0+dist)*(1.0+dist)) + B_1d / (1.0+dist)
    corr_d_component = B_13r * rho_d * rho_d * rho_d + B_12r * rho_d * rho_d + B_11r * rho_d
    corr_c_component = B_23r * rho_c * rho_c * rho_c + B_22r * rho_c * rho_c + B_21r * rho_c
    #return (B_1 + B_3d * dist*dist*dist + B_3r * rho*rho*rho + B_2d * dist*dist + B_2r * rho*rho + B_1d * dist + B_1r * rho)
    return B_1 + dist_component + corr_d_component +  corr_c_component

@vectorize([float64(float64, float64)], target ='cpu')
def hj_theta(t_ij, result):
    return t_ij * result

@numba.njit(cache = True, parallel=True)
def run(voxel_size, params, label, dist, hs, corr_d, corr_c):
    result = label
    for i in numba.prange(voxel_size):
        t_ij = theta_ij(dist[i], corr_d[i], corr_c[i], params[1], params[2], params[3], params[4],params[5], params[6], params[7],params[8], params[9], params[10])
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
def sample_using_workers(voxel_size, params, label, dist, epochs, hs, corr_d, corr_c):
    results = np.empty((epochs, voxel_size))
    for e in range(epochs):
        result = np.copy(label)
        for i in numba.prange(voxel_size):
            t_ij = theta_ij(dist[i], corr_d[i], corr_c[i], params[1], params[2], params[3], params[4],params[5], params[6], params[7],params[8], params[9], params[10])
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
def sample_ray_nocomp(voxel_size, params, label, dist, epochs, hs, corr_d, corr_c):
    theta = sample_using_workers(voxel_size, params, label, dist, epochs, hs, corr_d, corr_c)
    gamma = np.sum(theta, axis=0) # (epochs,1)
    return gamma

@ray.remote
def sample_ray(voxel_size, params, label, dist, mapping, H_reparam, epochs, hs, corr_d, corr_c):
    theta = sample_using_workers(voxel_size, params, label, dist, epochs, hs, corr_d, corr_c)

    H_iter = np.zeros((epochs, 11))
    H_mean = np.zeros(11)
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
            ##add corr
            H_iter[iteration][5] = tmp
                   

            H_iter[iteration][6] = tmp
                    

            H_iter[iteration][7] = tmp
                 

            H_iter[iteration][8] = tmp
               

            H_iter[iteration][9] = tmp
                 

            H_iter[iteration][10] = tmp
                   
        else:
            tmp = hi_dij(h_i, h_j) + 1 # 1/(1+d_ij)
            H_iter[iteration][2] = (-np.sum(tmp**(-1)))

            H_iter[iteration][3] = (-np.sum(tmp**(-2)))

            H_iter[iteration][4] = (-np.sum(tmp**(-3)))

            tmp = corr_d # corr_d
            H_iter[iteration][5] = (-np.sum(tmp**(3)))

            H_iter[iteration][6] = (-np.sum(tmp**(2)))
            H_iter[iteration][7] = (-np.sum(tmp**(1)))

            tmp = corr_c # corr_c
            H_iter[iteration][8] = (-np.sum(tmp**(3)))
            H_iter[iteration][9] = (-np.sum(tmp**(2)))
            H_iter[iteration][10] = (-np.sum(tmp**(1)))


        H_iter[iteration] /= H_reparam
        H_mean += H_iter[iteration]
        exp_sum[iteration] = -np.sum(params*H_iter[iteration])

        iteration += 1

    return (H_iter, H_mean, exp_sum, gamma)
 

def initial_ks(x_size, p, gamma):
    result = gamma
    for i in range(x_size):
        
        result[i] = np.random.binomial(1, p)
        if result[i] == 1:
            result[i] = result[i] * 100
        else:
            result[i] =1

    return result


def get_ks(x_size, gamma):
    ks = gamma
    for i in range(x_size):
    
        ks[i] = 100 * gamma[i]

    return ks


#https://nenadmarkus.com/p/all-pairs-euclidean/
def hi_dij(A, B):
    sqrA = np.broadcast_to(np.sum(np.power(A, 2), 1).reshape(A.shape[0], 1), (A.shape[0], B.shape[0]))
    sqrB = np.broadcast_to(np.sum(np.power(B, 2), 1).reshape(B.shape[0], 1), (B.shape[0], A.shape[0])).transpose()
    return np.sqrt(sqrA - 2*np.matmul(A, B.transpose()) + sqrB)

def parse_params(params_file):


    with open(params_file, 'r') as file:
        lines = file.readlines()
        params = np.array([float64(i.strip()) for i in lines[0:11]])
       
    return params

def back_to_full_dim_3d():
    mapping = np.load('1d_roi_to_3d_mapping.npy')
    lis = np.load('./result_gem_corr_gem/lis.npy')

    signal_full = np.zeros((94, 119, 98))

    for i in range(mapping.shape[0]):
        index = mapping[i].astype(int)
        a = index[0].astype(int)
        b = index[1].astype(int)
        c = index[2].astype(int)
        signal_full[a,b,c] = lis[i]


    np.save('./result_gem_corr_gem/lis_gem_full_dim.npy', signal_full)
    cropped = nib.Nifti1Image(signal_full, np.eye(4))
    cropped.header.get_xyzt_units()
    cropped.to_filename('./result_gem_corr_gem/lis_gem_full_dim.nii')


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

        dist = np.memmap(os.path.join(args.savepath, 'distance.npy'), dtype='float16', mode='r', shape=(model.size, model.size)).astype(np.float32)
        corr_c = np.memmap('/scratch/qj2022/ADNI_PET/corr_precompute_CN/correlation_CN_all.npy', dtype='float16', mode='r', shape=(model.size, model.size)).astype(np.float32)
        corr_d = np.memmap('/scratch/qj2022/ADNI_PET/corr_precompute_AD/correlation_AD_all.npy', dtype='float16', mode='r', shape=(model.size, model.size)).astype(np.float32) # -----------------------------ADD RAY
        
        global dist_id
        global corr_c_id
        global corr_d_id

        dist_id = ray.put(dist)
        corr_c_id = ray.put(corr_c)
        corr_d_id = ray.put(corr_d)

        model.gem()

        params_directory = os.path.join(args.savepath, 'result_gem_corr_gem/params.txt')
        params = parse_params(params_directory)

        ks = np.load(os.path.join(args.savepath, 'result_gem_corr_gem/ks.npy'))

  
        gamma = model.gibb_sampler_nocomp(burn_in=100, n_samples=1000, params=params, ks = ks, x=x)
        np.save(os.path.join(args.savepath, 'result_gem_corr_gem/gamma.npy'), gamma)
        model.p_lis(gamma)

        back_to_full_dim_3d()

    ray.shutdown()
