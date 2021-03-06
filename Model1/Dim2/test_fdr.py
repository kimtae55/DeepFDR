import os
import numpy as np
from test_data import Data
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import pandas as pd

class Model1:
    def __init__(self, x_file, rng_seed):
        #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        torch.backends.cudnn.deterministic = True
        
        self.SAVE_DIR = os.path.join(os.getcwd(), '../data/' + str(rng_seed) +'/result')
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
                
        self.KERNEL_PATH = os.path.join(os.getcwd(), 'kernel.txt')

        self.params = {'B': 0,
                       'H': 0,
                       #'B': 0.8,
                       #'H': -2.5,
                       'B_prev': 0,
                       'H_prev': 0,
                       'pL': torch.tensor([0.5,0.5]),
                       'muL': torch.tensor([-1.0,3.0]),
                       'sigmaL2': torch.tensor([1.0,1.0]),
                       #'pL': torch.tensor([0.5,0.5]),
                       #'muL': torch.tensor([-2.0,2.0]),
                       #'sigmaL2': torch.tensor([2.0,1.0]),
                       'pL_prev': torch.tensor([0.0,0.0]),
                       'muL_prev': torch.tensor([0.0,0.0]),
                       'sigmaL2_prev': torch.tensor([0.0,0.0])
                       }
        self.const = {'a': 1,  # for penalized likelihood for L >= 2
                      'b': 2,
                      'delta': 1e-3,
                      'maxIter': 500,
                      'eps1': 1e-3,
                      'eps2': 1e-3,
                      'eps3': 1e-4,
                      'alpha': 1e-4,
                      'burn_in': 1000, # 1000
                      'num_samples': 1000, # 5000
                      'L': 2,
                      'newton_max': 3,
                      'fdr_control': 0.1,
                      'tiny': 1e-8
                      }
        self.x = torch.from_numpy(np.loadtxt(x_file).reshape((Data.VOXEL_SIZE,Data.VOXEL_SIZE))).double().cuda()
        self.gamma = torch.zeros(self.x.shape).cuda()  # P(theta | x)
        self.init = torch.zeros(self.x.shape).cuda()  # initial theta value for gibb's sampling
        self.H_x = torch.zeros(2).cuda()
        self.H_mean = torch.zeros(2).cuda()
        self.log_sum = torch.zeros(1).cuda()
        self.U = torch.zeros(2).cuda()
        self.I_inv = torch.zeros((2, 2)).cuda()
        self.H_iter = torch.zeros((self.const['num_samples'], 2)).cuda()
        self.H_x_iter = torch.zeros((self.const['num_samples'], 2)).cuda()
        self.convergence_count = 0
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        init_prob = torch.full((Data.VOXEL_SIZE, Data.VOXEL_SIZE), 0.5)
        self.init = torch.bernoulli(init_prob).cuda()  # p = 0.5 to choose 1
        
        self.start = time.time()
        self.end = 0
        
        self.white_map = torch.from_numpy(np.indices((Data.VOXEL_SIZE, Data.VOXEL_SIZE)).sum(axis=0) % 2)
        self.white_map = self.white_map.type(torch.FloatTensor).cuda()
        ones = torch.ones((Data.VOXEL_SIZE, Data.VOXEL_SIZE)).cuda()
        self.black_map = torch.logical_xor(self.white_map, ones).type(torch.FloatTensor)
        self.black_map = self.black_map.cuda()

    def gem(self):
        '''
        E-step: Compute the following
        phi: (6), (7), (8)
        varphi: (10), (11)
        Iterate until convergence using delta (13)
        '''
        # compute first term in hs(t) = h(t) - .... equation-
        mu_0 = 0
        sigma0_sq = 1
        const_0 = torch.sqrt(torch.tensor([2 * torch.pi * sigma0_sq])).type(torch.DoubleTensor).cuda()
        ln_0 = (const_0 * torch.exp(torch.pow(self.x - mu_0, 2) / (2 * sigma0_sq))).type(torch.DoubleTensor).cuda()

        for t in range(self.const['maxIter']):
            print("")
            print("GEM iter: ", t)
            print("B:  " + str(self.params['B']) + " B_prev: " + str(self.params['B_prev']))
            print("H:  " + str(self.params['H']) + " H_prev: " + str(self.params['H_prev']))
            print("pL:  " + str(self.params['pL']) + " B_prev: " + str(self.params['pL_prev']))
            print("muL:  " + str(self.params['muL']) + " muL_prev: " + str(self.params['muL_prev']))
            print("sigmaL2:  " + str(self.params['sigmaL2']) + " sigmaL2_prev: " + str(self.params['sigmaL2_prev']))
            # varphi: (10), (11)
            self.params['B_prev'] = self.params['B']
            self.params['H_prev'] = self.params['H']
            for l in range(self.const['L']):
                self.params['muL_prev'][l] = self.params['muL'][l]
                self.params['sigmaL2_prev'][l] = self.params['sigmaL2'][l]
                self.params['pL_prev'][l] = self.params['pL'][l]
            
            gamma_sum = 0  # !!!!!!!!!!! need to compute

            # calculate hs(t)
            ln_1 = 0
            for l in range(self.const['L']):
                const_1 = ((self.params['pL'][l]) / (torch.sqrt(2 * torch.pi * self.params['sigmaL2'][l]))).type(torch.DoubleTensor).cuda()
                term_1 = torch.exp(-(torch.pow(self.x - self.params['muL'][l], 2)) / (2 * self.params['sigmaL2'][l])).type(torch.DoubleTensor).cuda()
                ln_1 +=  const_1 * term_1
            ln_term = ln_0*ln_1
            #print(ln_0)
            #print("ln_1:", ln_1)
            #print(ln_term)
            # Gibb's sampler to generate theta and theta_x
            theta, self.H_mean, self.H_iter, exp_sum, g = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], self.params['H'], self.params['B'], 1.0)
            theta_x, self.H_x, self.H_x_iter, exp_x_sum, self.gamma = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], self.params['H'], self.params['B'], ln_term)
            #print(self.gamma)

            # Monte carlo averages
            self.H_mean /= self.const['num_samples']
            self.H_x /= self.const['num_samples']
            self.gamma /= self.const['num_samples']
            gamma_sum = torch.sum(self.gamma)
            
            # compute log factor for current varphi
            # Because np.exp can overflow with large value of exp_sum, a modification is made
            # where we subtract by max_val. Same is done for log_sum_next
            self.log_sum = torch.zeros(1)
            #max_val = torch.amax(exp_sum)
            #self.log_sum = torch.sum(torch.exp(exp_sum - max_val))
            #self.log_sum = torch.log(self.log_sum) + max_val
            self.log_sum = torch.log(torch.sum(torch.exp(exp_sum.type(torch.DoubleTensor))))
            # compute U, I
            # U --> 2x1 matrix, I --> 2x2 matrix, trans(U)*inv(I)*U = 1x1 matrix
            self.U = self.H_x - self.H_mean
            self.I_inv = self.compute_I_inverse(self.const['num_samples'], self.H_iter, self.H_mean)
            # phi: (6), (7), (8)
            f_x = torch.zeros(self.x.shape).type(torch.DoubleTensor).cuda()
            omega_sum = torch.zeros(self.const['L']).type(torch.DoubleTensor).cuda()
            omega = torch.empty((2, Data.VOXEL_SIZE, Data.VOXEL_SIZE)).type(torch.DoubleTensor).cuda()

########################################### CHECK ALL THESE EQUATIONS
            for l in range(self.const['L']):
                f_x += self.params['pL'][l] * self.norm(self.x, self.params['muL'][l], self.params['sigmaL2'][l])
            for l in range(self.const['L']):
                omega[l] = self.gamma * self.params['pL'][l] / f_x \
                                       * self.norm(self.x, self.params['muL'][l], self.params['sigmaL2'][l])
                omega_sum[l] = torch.sum(omega[l])

            # t+1 update
            for l in range(self.const['L']):
                self.params['muL'][l] = (torch.sum(omega[l] * self.x)) / (omega_sum[l])
                self.params['sigmaL2'][l] = (2.0 * self.const['a'] + self.sum_sigmaL2(omega[l], self.x, self.params['muL'][l])) / (2.0 * self.const['b'] + omega_sum[l])
                self.params['pL'][l] = (omega_sum[l]) / (gamma_sum)
            
            # find varphi(t+1) that satisfies Armijo condition and compute Q2(t+1) - Q2(t)
            self.computeArmijo()
            
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
            '''
            # For extra visualization
            visualize = self.gamma.cpu().numpy()
            plt.imshow(visualize)
            plt.title('label')
            plt.show()
            '''
        # Save B, H, mu, sigma, pl, gamma
        params_directory = os.path.join(self.SAVE_DIR, 'params.txt')
        with open(params_directory, 'w') as outfile:
            for key, value in self.params.items():
                outfile.write(str(key) + ': ' + str(value) + '\n')

        gamma_directory = os.path.join(self.SAVE_DIR, 'gamma.txt')
        with open(gamma_directory, 'w') as outfile:
            for data_slice in self.gamma.cpu().numpy():
                np.savetxt(outfile, data_slice, fmt='%-8.4f')
                outfile.write('# New z slice\n')

    def computeArmijo(self):
        lambda_m = 1
        diff = torch.zeros(2)
        # eq(11) satisfaction condition
        while True:
            delta = torch.matmul(self.I_inv, self.U)
            #print(delta)
            self.params['B'] = self.params['B_prev'] + lambda_m * delta[0]
            self.params['H'] = self.params['H_prev'] + lambda_m * delta[1]
            #print(self.params)

            # check for alternative criterion when delta_varphi is too small for armijo convergence
            # In practice, the Armijo condition (11) might not be satisfied
            # when the step length is too small
            diff[0] = torch.abs(torch.tensor([self.params['B'] - self.params['B_prev']])) / (
                        torch.abs(torch.tensor([self.params['B_prev']])) + self.const['eps1'])
            diff[1] = torch.abs(torch.tensor([self.params['H'] - self.params['H_prev']])) / (
                        torch.abs(torch.tensor([self.params['H_prev']])) + self.const['eps1'])
            max = torch.amax(diff)
            if max < self.const['eps3']:
                # alternative criterion
                break
            
            armijo = self.const['alpha'] * lambda_m * (torch.matmul(torch.transpose(self.U, 0, -1), torch.matmul(self.I_inv, self.U)))
            # use Gibbs sampler again to compute log_factor for delta_Q2 approximation
            log_factor = self.computeLogFactor()

            deltaQ2 = (self.params['B'] - self.params['B_prev']) * self.H_x[0] \
                      + (self.params['H'] - self.params['H_prev']) * self.H_x[1] + log_factor

            if deltaQ2 >= armijo:
                break
            else:
                lambda_m /= 2
        return

    def computeLogFactor(self):
        theta_next, H_next_mean, H_next_iter, exp_sum_next, g = self.gibb_sampler(self.const['burn_in'], self.const['num_samples'], self.params['H'], self.params['B'], 1.0)
        # compute log sum for varphi_next
        #max_val = torch.amax(exp_sum_next)
        #log_sum_next = torch.sum(torch.exp(exp_sum_next - max_val))
        #log_sum_next = torch.log(log_sum_next) + max_val
        log_sum_next = torch.log(torch.sum(torch.exp(exp_sum_next.type(torch.DoubleTensor))))
        result = log_sum_next - self.log_sum
        #print("result", result)
        return result

    def converged(self):
        # we have 8 parameters
        # B, H, pl[0], pl[1], mul[0], mul[1], sigmal[0], sigmal[1]
        diff = torch.zeros(8)
        diff[0] = torch.abs(torch.tensor([self.params['B']-self.params['B_prev']])) / (torch.abs(torch.tensor([self.params['B_prev']])) + self.const['eps1'])
        diff[1] = torch.abs(torch.tensor([self.params['H']-self.params['H_prev']])) / (torch.abs(torch.tensor([self.params['H_prev']])) + self.const['eps1'])
        diff[2] = torch.abs(torch.tensor([self.params['pL'][0]-self.params['pL_prev'][0]])) / (torch.abs(torch.tensor([self.params['pL_prev'][0]])) + self.const['eps1'])
        diff[3] = torch.abs(torch.tensor([self.params['pL'][1]-self.params['pL_prev'][1]])) / (torch.abs(torch.tensor([self.params['pL_prev'][1]])) + self.const['eps1'])
        diff[4] = torch.abs(torch.tensor([self.params['muL'][0]-self.params['muL_prev'][0]])) / (torch.abs(torch.tensor([self.params['muL_prev'][0]])) + self.const['eps1'])
        diff[5] = torch.abs(torch.tensor([self.params['muL'][1]-self.params['muL_prev'][1]])) / (torch.abs(torch.tensor([self.params['muL_prev'][1]])) + self.const['eps1'])
        diff[6] = torch.abs(torch.tensor([self.params['sigmaL2'][0]-self.params['sigmaL2_prev'][0]])) / (torch.abs(torch.tensor([self.params['sigmaL2_prev'][0]])) + self.const['eps1'])
        diff[7] = torch.abs(torch.tensor([self.params['sigmaL2'][1]-self.params['sigmaL2_prev'][1]])) / (torch.abs(torch.tensor([self.params['sigmaL2_prev'][1]])) + self.const['eps1'])
        max = torch.amax(diff)
        if max < self.const['eps2']:
            return True

        return False
    
    def gibb_sampler(self, burn_in, n_samples, h, beta, ln_term):

        # initialize labels
        init_prob = torch.full((Data.VOXEL_SIZE, Data.VOXEL_SIZE), 0.5)
        theta = torch.bernoulli(init_prob).cuda()  # p = 0.5 to choose 1
        kernel = torch.from_numpy(np.loadtxt(self.KERNEL_PATH).reshape((3,3))).float()
        kernel = torch.unsqueeze(kernel, 0)
        kernel = torch.unsqueeze(kernel, 0)
        kernel = kernel.cuda()
        H_iter = torch.zeros((self.const['num_samples'], 2)).cuda()
        H_mean = torch.zeros(2).cuda()
        exp_sum = torch.zeros(self.const['num_samples']).cuda()
        gamma = torch.zeros(self.x.shape).cuda()
        
        iteration = 0
        iter_burn = 0
        while iteration < n_samples:
            # calculate sum_nn
            # sum of neighboring pixels is a convolution operation
            theta = torch.unsqueeze(theta, 0)
            theta = torch.unsqueeze(theta, 0)
            sum_nn = torch.nn.functional.conv2d(theta, kernel, stride=1, padding=1).squeeze()
            # calculate exp_sum_nn
            numerator = torch.exp(sum_nn * beta + h)
            conditional_distribution = numerator / (1 + numerator)
            # then put it into bernoulli

            white = torch.bernoulli(conditional_distribution)
            white = torch.logical_and(white, self.white_map).type(torch.FloatTensor).cuda()

            black_label = torch.logical_and(theta.squeeze(), self.black_map).type(torch.FloatTensor).cuda()
            theta = torch.logical_or(black_label, white).type(torch.FloatTensor).cuda()
            theta = torch.unsqueeze(theta, 0)
            theta = torch.unsqueeze(theta, 0)
            sum_nn = torch.nn.functional.conv2d(theta, kernel, stride=1, padding=1).squeeze()

            # calculate exp_sum_nn
            numerator = torch.exp(sum_nn * beta + h)
            conditional_distribution = numerator / (1 + numerator)
            # then put it into bernoulli
            black = torch.bernoulli(conditional_distribution)
            black = torch.logical_and(black, self.black_map).type(torch.FloatTensor).cuda()
            theta = torch.logical_or(black, white).type(torch.FloatTensor).cuda()

            if iter_burn < burn_in:
                iter_burn = iter_burn + 1
                #print("count = : ", torch.count_nonzero(theta) / (Data.VOXEL_SIZE * Data.VOXEL_SIZE * Data.VOXEL_SIZE))
                #print("burn in: ", iter_burn)
            else:
                H_mean[0] += torch.sum(theta * sum_nn) #******************************* CHECK CORRECTNESS IN THESE SIGMAS
                H_mean[1] += torch.sum(theta)
                H_iter[iteration][0] = torch.sum(theta * sum_nn)
                H_iter[iteration][1] = torch.sum(theta)

                exp_sum[iteration] = -1 * self.H_iter[iteration][0] * self.params['B'] - self.H_iter[iteration][1] * self.params['H']
                gamma += theta
                iteration += 1

                
        return theta, H_mean, H_iter, exp_sum, gamma


    def compute_I_inverse(self, num_samples, H, H_mean):
        I = torch.zeros((2, 2)).cuda()
        I_inv = torch.zeros((2, 2)).cuda()
        for i in range(num_samples):
            I[0][0] += (H[i][0] - H_mean[0]) * (H[i][0] - H_mean[0])
            I[0][1] += (H[i][1] - H_mean[1]) * (H[i][0] - H_mean[0])
            I[1][0] += (H[i][1] - H_mean[1]) * (H[i][0] - H_mean[0])
            I[1][1] += (H[i][1] - H_mean[1]) * (H[i][1] - H_mean[1])

        I /= num_samples - 1  # n-1
        determinant = I[1][1] * I[0][0] - I[0][1] * I[1][0]
        if determinant > 0:
            I_inv = torch.inverse(I)
        else:
            # identity matrix
            # check if this is happening often
            I_inv = torch.eye(2).cuda()

        #print("I_inv: ", I_inv)
        return I_inv
        
    def norm(self, x, mu, sigma2):
        return torch.exp(-1 * (x - mu) * (x - mu) / (2 * sigma2)) / (torch.sqrt(2 * torch.pi * sigma2))

    def sum_sigmaL2(self, omega, x, muL):       
        return torch.sum(omega * torch.pow(x - muL, 2))
        
        # uses cpu since it doesn't take long            
    def p_lis(self, gamma_1, label):
        # LIS = P(theta = 0 | x)
        # gamma_1 = P(theta = 1 | x) = 1 - LIS
        dtype = [('index', float), ('value', float)]
        lis = np.zeros((Data.VOXEL_SIZE * Data.VOXEL_SIZE), dtype=dtype)
        for vx in range(Data.VOXEL_SIZE):
            for vy in range(Data.VOXEL_SIZE):
                index =  (vy * Data.VOXEL_SIZE) + vx
                lis[index]['index'] = index
                # can't just do this 
                lis[index]['value'] = 1 - gamma_1[vx][vy]
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
        print("k", k)

        signal_lis = np.ones((Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        for j in range(k):
            index = lis[j]['index']
            vy = index // Data.VOXEL_SIZE  # integer division
            vx = index % Data.VOXEL_SIZE
            vy = int(vy)
            vx = int(vx)
            signal_lis[vx][vy] = 0  # reject these voxels, rest are 1

        # Compute FDR, FNR, ATP using LIS and Label
        # FDR -> (theta = 0) / num_rejected 
        # FNR -> (theta = 1) / num_not_rejected
        # ATP -> (theta = 1) that is rejected
        num_rejected = k
        num_not_rejected = (Data.VOXEL_SIZE*Data.VOXEL_SIZE*Data.VOXEL_SIZE) - k
        fdr = 0
        fnr = 0
        atp = 0
        for i in range(Data.VOXEL_SIZE):
            for j in range(Data.VOXEL_SIZE):
                if signal_lis[i][j] == 0: # rejected
                    if label[i][j] == 0:
                        fdr += 1
                    elif label[i][j] == 1: 
                        atp += 1
                elif signal_lis[i][j] == 1: # not rejected
                    if label[i][j] == 1:
                        fnr += 1
        fdr /= num_rejected
        fnr /= num_not_rejected          
        
        # Save final signal_file
        signal_directory = os.path.join(self.SAVE_DIR, 'signal.txt')
        with open(signal_directory, 'w') as outfile:
            outfile.write('fdr: ' + str(fdr) + '\n')
            outfile.write('fnr: ' + str(fnr) + '\n')
            outfile.write('atp: ' + str(atp) + '\n')
            for data_slice in signal_lis:
                np.savetxt(outfile, data_slice, fmt='%-8.4f')
                outfile.write('# New z slice\n')
               
        return
        
if __name__ == "__main__":
    rng_seed = sys.argv[1]
    print("RAND: ", rng_seed)
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    print("GPU: ", torch.cuda.get_device_name(0))
    fdr = Model1('../data/' + str(rng_seed) +'/x_val.txt', rng_seed)
    fdr.gem()
    gamma_file = '../data/' + str(rng_seed) +'/result' +'/gamma.txt'
    label_file = '../data/' + str(rng_seed) +'/label/label.txt'
    fdr.p_lis(np.loadtxt(gamma_file).reshape((Data.VOXEL_SIZE,Data.VOXEL_SIZE)),
             np.loadtxt(label_file).reshape((Data.VOXEL_SIZE,Data.VOXEL_SIZE)))



