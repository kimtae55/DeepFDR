import numpy as np
import os
from DataUtil import Data
import math
from scipy.stats import bernoulli
import time

class Util:
    @staticmethod
    def norm(x, mu, sigma2):
        return math.exp(-1 * (x - mu) * (x - mu) / (2 * sigma2)) / (math.sqrt(2 * math.pi * sigma2))

    @staticmethod
    def in_range(i, j, k, size):
        return (0 <= i < size) and (0 <= j < size) and (0 <= k < size)

    @staticmethod
    def sum_muL(omega, x):
        sum = 0
        for vx in range(Data.VOXEL_SIZE):
            for vy in range(Data.VOXEL_SIZE):
                for vk in range(Data.VOXEL_SIZE):
                    sum += omega[vx][vy][vk] * x[vx][vy][vk]
        return sum

    @staticmethod
    def sum_sigmaL2(omega, x, muL):
        sum = 0
        for vx in range(Data.VOXEL_SIZE):
            for vy in range(Data.VOXEL_SIZE):
                for vk in range(Data.VOXEL_SIZE):
                    sum += omega[vx][vy][vk] * ((x[vx][vy][vk] - muL) ** 2)
        return sum

    @staticmethod
    def model1_eq1(B, h, i, j, k, theta, size):
        sum = 0
        if Util.in_range(i, j, k + 1, size):
            sum += theta[i][j][k + 1]
        if Util.in_range(i, j, k - 1, size):
            sum += theta[i][j][k - 1]
        if Util.in_range(i, j + 1, k, size):
            sum += theta[i][j + 1][k]
        if Util.in_range(i, j - 1, k, size):
            sum += theta[i][j - 1][k]
        if Util.in_range(i + 1, j, k, size):
            sum += theta[i + 1][j][k]
        if Util.in_range(i - 1, j, k, size):
            sum += theta[i - 1][j][k]
        num = math.exp((B * sum + h))  # probability of theta = 1, so take out theta_s from model1_eq1
        denom = 1 + math.exp(B * sum + h)
        return (num / denom)

    @staticmethod
    def sum_neighbors(i, j, k, theta, size):
        sum = 0
        if Util.in_range(i, j, k + 1, size):
            sum += theta[i][j][k + 1]
        if Util.in_range(i, j, k - 1, size):
            sum += theta[i][j][k - 1]
        if Util.in_range(i, j + 1, k, size):
            sum += theta[i][j + 1][k]
        if Util.in_range(i, j - 1, k, size):
            sum += theta[i][j - 1][k]
        if Util.in_range(i + 1, j, k, size):
            sum += theta[i + 1][j][k]
        if Util.in_range(i - 1, j, k, size):
            sum += theta[i - 1][j][k]
        return sum

    @staticmethod
    def compute_I_inverse(num_samples, H, H_mean):
        I = np.zeros((2, 2))
        for i in range(num_samples):
            I[0][0] += (H[i][0] - H_mean[0]) * (H[i][0] - H_mean[0])
            I[0][1] += (H[i][1] - H_mean[1]) * (H[i][0] - H_mean[0])
            I[1][0] += (H[i][1] - H_mean[1]) * (H[i][0] - H_mean[0])
            I[1][1] += (H[i][1] - H_mean[1]) * (H[i][1] - H_mean[1])

        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                I[i][j] /= num_samples - 1  # n-1

        I_inv = np.linalg.inv(I)
        return I_inv


class Model1:
    def __init__(self, x_file):
        self.params = {'B': 0,
                       'H': 0,
                       'B_prev': 0,
                       'H_prev': 0,
                       'pL': np.array([0.3,0.7]),
                       'muL': np.array([1,1.5]),
                       'sigmaL2': np.array([1.5,1.5]),
                       'pl_prev': np.zeros(2),
                       'muL_prev': np.zeros(2),
                       'sigmaL2_prev': np.zeros(2)
                       }
        self.const = {'a': 1,  # for penalized likelihood for L >= 2
                      'b': 2,
                      'delta': 1e-3,
                      'maxIter': 1000,
                      'eps1': 1e-3,
                      'eps2': 1e-3,
                      'eps3': 1e-4,
                      'alpha': 1e-4,
                      'burn_in': 100, # 1000
                      'num_samples': 32, # 5000
                      'L': 2,
                      'newton_max': 3,
                      'fdr_control': 0.1,
                      'tiny': 1e-8
                      }
        self.x = Data.loadX(os.path.join(os.getcwd(), x_file))
        self.gamma = np.zeros(self.x.shape)  # P(theta | x)
        self.init = np.zeros(self.x.shape)  # initial theta value for gibb's sampling
        self.H_x = np.zeros(2)
        self.H_mean = np.zeros(2)
        self.log_sum = np.zeros(1)
        self.U = np.zeros(2)
        self.I_inv = np.zeros((2, 2))
        self.H = np.zeros((self.const['num_samples'], 2))
        self.convergence_count = 0
        for i in range(Data.VOXEL_SIZE):
            for j in range(Data.VOXEL_SIZE):
                for k in range(Data.VOXEL_SIZE):
                    self.init[i][j][k] = bernoulli.rvs(0.5)  # p = 0.5 to choose 1
        self.start = time.time()
        self.end = 0

    def gem(self):
        '''
        E-step: Compute the following
        phi: (6), (7), (8)
        varphi: (10), (11)
        Iterate until convergence using delta (13)
        '''
        np.random.seed(12345)

        # compute first term in hs(t) = h(t) - .... equation
        mu_0 = 0
        sigma0_sq = 1

        const_0 = 1 / (math.sqrt(2 * math.pi * sigma0_sq))
        ln_0 = np.zeros(self.x.shape)
        for vx in range(Data.VOXEL_SIZE):
            for vy in range(Data.VOXEL_SIZE):
                for vk in range(Data.VOXEL_SIZE):
                    ln_0[vx][vy][vk] = math.log(
                        const_0 * math.exp(-((self.x[vx][vy][vk] - mu_0) ** 2) / (2 * sigma0_sq)))

        # GEM loop
        for t in range(self.const['maxIter']):
            print("GEM iter: ", t)
            print(self.params)
            # varphi: (10), (11)
            self.params['B_prev'] = self.params['B']
            self.params['H_prev'] = self.params['H']
            gamma_sum = 0  # !!!!!!!!!!! need to compute

            # calculate hs(t)
            hs = np.zeros(self.x.shape)
            for vx in range(Data.VOXEL_SIZE):
                for vy in range(Data.VOXEL_SIZE):
                    for vk in range(Data.VOXEL_SIZE):
                        ln_1 = 0
                        for l in range(self.const['L']):
                            ln_1 += ((self.params['pL'][l]) / (math.sqrt(2 * math.pi * self.params['sigmaL2'][l]))) * \
                                    math.exp(-(((self.x[vx][vy][vk] - self.params['muL'][l]) ** 2)) / (2 * self.params['sigmaL2'][l]))
                        #print(ln_1)
                        ln_1 = math.log(ln_1)
                        hs[vx][vy][vk] = self.params['H'] - ln_0[vx][vy][vk] + ln_1

            # Gibb's sampler to generate theta and theta_x
            exp_sum = np.zeros(self.const['num_samples'])
            theta = self.init
            theta_x = self.init
            iteration = 0
            iter_burn = 0
            while iteration < self.const['num_samples']:
                # do stuff
                for vx in range(Data.VOXEL_SIZE):
                    for vy in range(Data.VOXEL_SIZE):
                        for vk in range(Data.VOXEL_SIZE):
                            conditional_distribution_x = Util.model1_eq1(self.params['B'], hs[vx][vy][vk],
                                                                       vx, vy, vk, theta_x, Data.VOXEL_SIZE)
                            theta_x[vx][vy][vk] = bernoulli.rvs(conditional_distribution_x)   
                            #print("p, theta: ", conditional_distribution_x, theta_x[vx][vy][vk])
                if iter_burn < self.const['burn_in']:
                    iter_burn += 1
                else:
                    for vx in range(Data.VOXEL_SIZE):
                        for vy in range(Data.VOXEL_SIZE):
                            for vk in range(Data.VOXEL_SIZE):
                                self.H_x[0] += theta_x[vx][vy][vk] * Util.sum_neighbors(vx, vy, vk, theta_x,
                                                                                        Data.VOXEL_SIZE)
                                self.H_x[1] += theta_x[vx][vy][vk]
                                self.gamma[vx][vy][vk] += theta_x[vx][vy][vk]  # probability that theta = 1
                    iteration += 1 

            iteration = 0
            iter_burn = 0
            while iteration < self.const['num_samples']:
                # do stuff
                for vx in range(Data.VOXEL_SIZE):
                    for vy in range(Data.VOXEL_SIZE):
                        for vk in range(Data.VOXEL_SIZE):
                            conditional_distribution = Util.model1_eq1(self.params['B'], self.params['H'],
                                                                       vx, vy, vk, theta, Data.VOXEL_SIZE)
                            theta[vx][vy][vk] = bernoulli.rvs(conditional_distribution)
                            # print("p, theta: ", conditional_distribution, theta[vx][vy][vk])
                if iter_burn < self.const['burn_in']:
                    iter_burn += 1
                else:
                    for vx in range(Data.VOXEL_SIZE):
                        for vy in range(Data.VOXEL_SIZE):
                            for vk in range(Data.VOXEL_SIZE):
                                sum = Util.sum_neighbors(vx, vy, vk, theta, Data.VOXEL_SIZE)
                                self.H_mean[0] += theta[vx][vy][vk] * sum
                                self.H_mean[1] += theta[vx][vy][vk]

                                self.H[iteration][0] += theta[vx][vy][vk] * sum
                                self.H[iteration][1] += theta[vx][vy][vk]
                    exp_sum[iteration] = -1 * self.H[iteration][0] * self.params['B'] - self.H[iteration][1] * self.params['H']
                    iteration += 1

            # Monte carlo averages
            self.H_mean[0] /= self.const['num_samples']
            self.H_mean[1] /= self.const['num_samples']
            self.H_x[0] /= self.const['num_samples']
            self.H_x[1] /= self.const['num_samples']
            for vx in range(Data.VOXEL_SIZE):
                for vy in range(Data.VOXEL_SIZE):
                    for vk in range(Data.VOXEL_SIZE):
                        self.gamma[vx][vy][vk] /= self.const['num_samples']
                        gamma_sum += self.gamma[vx][vy][vk]
            # compute log factor for current varphi
            # Because np.exp can overflow with large value of exp_sum, a modification is made
            # where we subtract by max_val. Same is done for log_sum_next
            self.log_sum = np.zeros(1)
            max_val = np.amax(exp_sum)
            for i in range(self.const['num_samples']):
                self.log_sum += np.exp(exp_sum[i] - max_val)
            self.log_sum = max_val + np.log(self.log_sum)
            print("self.log_sum:", self.log_sum)

            # compute U, I
            # U --> 2x1 matrix, I --> 2x2 matrix, trans(U)*inv(I)*U = 1x1 matrix
            print("H_x: ", self.H_x)
            print("H_mean: ", self.H_mean)
            self.U[0] = self.H_x[0] - self.H_mean[0]
            self.U[1] = self.H_x[1] - self.H_mean[1]
            self.I_inv = Util.compute_I_inverse(self.const['num_samples'], self.H, self.H_mean)

            # phi: (6), (7), (8)
            self.params['muL_prev'] = self.params['muL']
            self.params['sigmaL2_prev'] = self.params['sigmaL2']
            self.params['pL_prev'] = self.params['pL']

            f_x = np.zeros(self.x.shape)
            omega_sum = np.zeros(self.const['L'])
            omega = np.empty((2, Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))

            for vx in range(Data.VOXEL_SIZE):
                for vy in range(Data.VOXEL_SIZE):
                    for vk in range(Data.VOXEL_SIZE):
                        for l in range(self.const['L']):
                            f_x[vx][vy][vk] += self.params['pL'][l] \
                                               * Util.norm(self.x[vx][vy][vk],
                                                           self.params['muL'][l], self.params['sigmaL2'][l])
                        for l in range(self.const['L']):
                            omega[l][vx][vy][vk] = self.gamma[vx][vy][vk] * self.params['pL'][l] / f_x[vx][vy][vk] \
                                                   * Util.norm(self.x[vx][vy][vk],
                                                               self.params['muL'][l], self.params['sigmaL2'][l])
                            omega_sum[l] += omega[l][vx][vy][vk]

            # t+1 update
            for l in range(self.const['L']):
                self.params['muL'][l] = Util.sum_muL(omega[l], self.x) / omega_sum[l]
                self.params['sigmaL2'][l] = (2 * self.const['a'] + Util.sum_sigmaL2(omega[l], self.x,
                                                                                    self.params['muL'][l])) \
                                            / (2 * self.const['b'] + omega_sum[l])
                self.params['pL'][l] = omega_sum[l] / gamma_sum

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

        # Save B, H, mu, sigma, pl, gamma
        with open('params.txt', 'w') as outfile:
            for key, value in self.params.items():
                outfile.write(str(key) + ': ' + str(value) + '\n')

        with open('gamma.txt', 'w') as outfile:
            for data_slice in self.gamma:
                np.savetxt(outfile, data_slice, fmt='%-8.4f')
                outfile.write('# New z slice\n')

    def p_lis(self, gamma_1):
        # LIS = P(theta = 0 | x)
        # gamma_1 = P(theta = 1 | x) = 1 - LIS
        dtype = [('index', float), ('value', float)]
        lis = np.zeros((Data.VOXEL_SIZE * Data.VOXEL_SIZE * Data.VOXEL_SIZE), dtype=dtype)
        for vx in range(Data.VOXEL_SIZE):
            for vy in range(Data.VOXEL_SIZE):
                for vk in range(Data.VOXEL_SIZE):
                    index = (vk * Data.VOXEL_SIZE * Data.VOXEL_SIZE) + (vy * Data.VOXEL_SIZE) + vx
                    lis[index]['index'] = index
                    lis[index]['value'] = 1 - gamma_1[vx][vy][vk]
        # sort using lis values
        np.sort(lis, order='value')

        # Data driven LIS-based FDR procedure
        sum = 0
        k = 0
        signal_lis = np.ones((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
        for j in range(len(lis)):
            sum += lis[j]['value']
            if sum > (j+1)*self.const['fdr_control']:
                k = j
                break

        for j in range(k):
            index = lis[j]['index']
            vk = index // (Data.VOXEL_SIZE*Data.VOXEL_SIZE)  # integer division
            index -= vk*Data.VOXEL_SIZE*Data.VOXEL_SIZE
            vy = index // Data.VOXEL_SIZE  # integer division
            vx = index % Data.VOXEL_SIZE
            signal_lis[vx][vy][vk] = 0  # reject these voxels, rest are 1

        # Save final signal_file
        with open('signal.txt', 'w') as outfile:
            for data_slice in signal_lis:
                np.savetxt(outfile, data_slice, fmt='%-8.4f')
                outfile.write('# New z slice\n')

        return

    def computeArmijo(self):
        lambda_m = 1
        diff = np.zeros(2)
        # eq(11) satisfaction condition
        while True:
            delta = np.matmul(self.I_inv, self.U)
            #print(delta)
            self.params['B'] = self.params['B_prev'] + lambda_m * delta[0]
            self.params['H'] = self.params['H_prev'] + lambda_m * delta[1]
            #print(self.params)

            # check for alternative criterion when delta_varphi is too small for armijo convergence
            # In practice, the Armijo condition (11) might not be satisfied
            # when the step length is too small
            diff[0] = np.fabs(self.params['B'] - self.params['B_prev']) / (
                        np.fabs(self.params['B_prev']) + self.const['eps1'])
            diff[1] = np.fabs(self.params['H'] - self.params['H_prev']) / (
                        np.fabs(self.params['H_prev']) + self.const['eps1'])
            max = np.amax(diff)
            if max < self.const['eps3']:
                print("alternative condition")
                break

            armijo = self.const['alpha'] * lambda_m * (np.matmul(np.transpose(self.U), np.matmul(self.I_inv, self.U)))
            # use Gibbs sampler again to compute log_factor for delta_Q2 approximation
            log_factor = self.computeLogFactor()

            deltaQ2 = (self.params['B'] - self.params['B_prev']) * self.H_x[0] \
                      + (self.params['H'] - self.params['H_prev']) * self.H_x[1] + log_factor

            if deltaQ2 >= armijo:
                break
            else:
                lambda_m /= 2

    def computeLogFactor(self):
        # Gibb's sampler to generate theta and theta_x
        H_next = np.zeros((self.const['num_samples'], 2))
        theta = self.init
        iteration = 0
        iter_burn = 0
        log_sum_next = np.zeros(1)
        exp_sum_next = np.zeros(self.const['num_samples'])
        while iteration < self.const['num_samples']:
            # do stuff
            for vx in range(Data.VOXEL_SIZE):
                for vy in range(Data.VOXEL_SIZE):
                    for vk in range(Data.VOXEL_SIZE):
                        conditional_distribution = Util.model1_eq1(self.params['B'], self.params['H'],
                                                                   vx, vy, vk, theta, Data.VOXEL_SIZE)
                        theta[vx][vy][vk] = bernoulli.rvs(conditional_distribution)

            if iter_burn < self.const['burn_in']:
                iter_burn += 1
            else:
                for vx in range(Data.VOXEL_SIZE):
                    for vy in range(Data.VOXEL_SIZE):
                        for vk in range(Data.VOXEL_SIZE):
                            sum = Util.sum_neighbors(vx, vy, vk, theta, Data.VOXEL_SIZE)

                            H_next[iteration][0] += theta[vx][vy][vk] * sum
                            H_next[iteration][1] += theta[vx][vy][vk]
                exp_sum_next[iteration] = -1* H_next[iteration][0] * self.params['B'] - H_next[iteration][1] * self.params['H']
                iteration += 1
        # compute log sum for varphi_next
        max_val = np.amax(exp_sum_next)
        for i in range(self.const['num_samples']):
        # CHANGE TO NP.EXP OR TORCH.EXP , MATH.exp has decimal point restrictions
        # Re-structure the tensors such that the operations can be done in parallel 
            log_sum_next += np.exp(exp_sum_next[i] - max_val)
        log_sum_next = max_val + np.log(log_sum_next)
        result = log_sum_next - self.log_sum
        #print("result", result)
        return result

    def converged(self):
        # we have 8 parameters
        # B, H, pl[0], pl[1], mul[0], mul[1], sigmal[0], sigmal[1]
        diff = np.zeros(8)
        diff[0] = math.fabs(self.params['B']-self.params['B_prev']) / (math.fabs(self.params['B_prev']) + self.const['eps1'])
        diff[1] = math.fabs(self.params['H']-self.params['H_prev']) / (math.fabs(self.params['H_prev']) + self.const['eps1'])
        diff[2] = math.fabs(self.params['pL'][0]-self.params['pl_prev'][0]) / (math.fabs(self.params['pl_prev'][0]) + self.const['eps1'])
        diff[3] = math.fabs(self.params['pL'][1]-self.params['pl_prev'][1]) / (math.fabs(self.params['pl_prev'][1]) + self.const['eps1'])
        diff[4] = math.fabs(self.params['muL'][0]-self.params['muL_prev'][0]) / (math.fabs(self.params['muL_prev'][0]) + self.const['eps1'])
        diff[5] = math.fabs(self.params['muL'][1]-self.params['muL_prev'][1]) / (math.fabs(self.params['muL_prev'][1]) + self.const['eps1'])
        diff[6] = math.fabs(self.params['sigmaL2'][0]-self.params['sigmaL2_prev'][0]) / (math.fabs(self.params['sigmaL2_prev'][0]) + self.const['eps1'])
        diff[7] = math.fabs(self.params['sigmaL2'][1]-self.params['sigmaL2_prev'][1]) / (math.fabs(self.params['sigmaL2_prev'][1]) + self.const['eps1'])

        max = np.amax(diff)
        if max < self.const['eps2']:
            return True

        return False


if __name__ == "__main__":
    #gamma_1 = np.loadtxt("../gamma.txt").reshape((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
    #fdr.p_lis(gamma_1)
    #gamma = np.zeros((Data.VOXEL_SIZE, Data.VOXEL_SIZE, Data.VOXEL_SIZE))
    fdr = Model1("../data_15x15x15/x_val.txt")
    fdr.gem()


