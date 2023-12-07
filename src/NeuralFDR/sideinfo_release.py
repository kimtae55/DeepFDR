import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from numpy import array
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans
    
    
def p_value_beta_fit(p, lamb=0.8, bin_num=50, vis=0):
    pi_0=np.divide(np.sum(p>lamb), p.shape[0] * (1-lamb))
    temp_p=np.zeros([0])
    step_size=np.divide(1,np.float(bin_num))
    fil_num=np.int(np.divide(pi_0*p.shape[0],bin_num))+1
    for i in range(bin_num):
        p1=p[p>step_size*(i-1)]
        p1=p1[p1 <= step_size*i]
        choice_num= np.max(p1.shape[0] - fil_num,0)
        if choice_num > 1:
            choice=np.random.choice(p1.shape[0], choice_num)
            temp_p=np.concatenate([temp_p,p1[choice]]).T
    if vis==1:
        plt.figure()
        plt.hist(temp_p, bins=100, normed=True)       
    a, b, loc, scale = beta.fit(temp_p,floc=0,fscale=1)
    return pi_0, a, b

def beta_mixture_pdf(x,pi_0,a,b):
    return beta.pdf(x,a,b)*(1-pi_0)+pi_0

def Storey_BH(x, alpha = 0.05, lamb=0.4, n = None):
    pi0_hat=np.divide(np.sum(x>lamb),x.shape[0] *(1-lamb))
    alpha /= pi0_hat
    x_s = sorted(x)
    if n is None:
        n = len(x_s)
    ic = 0
    for i in range(len(x_s)):
        if x_s[i] < i*alpha/float(n):
            ic = i
    return ic, x_s[ic], pi0_hat

def Opt_t_cal_discrete(p, X, num_case=2,step_size=0.0001,ub=0.05,n_samples=2000,alpha=0.05):
    # Fit the beta mixture parameters
    fit_param=np.zeros([num_case, 3])
    for i in range(num_case):
        fit_param[i,:]=p_value_beta_fit(p[X==i])

    # Calculating the ratios 
    t_opt=np.zeros([num_case])
    max_idx=np.argmin(fit_param[:,0])
    x_grid = np.arange(0, ub, step_size)
    t_ratio=np.zeros([num_case,x_grid.shape[0]])
    for i in range(num_case):
        t_ratio[i,:] = np.divide(beta_mixture_pdf(
            x_grid,fit_param[i,0],fit_param[i,1],fit_param[i,2]), fit_param[i,0])

    # Increase the threshold
    for i in range(len(x_grid)):
        t=np.zeros([num_case])
        # undate the search optimal threshold
        t[max_idx]=x_grid[i]
        c=t_ratio[max_idx,i]
        for j in range(num_case):
            if j != max_idx: 
                for k in range(len(x_grid)):
                    if k == len(x_grid)-1:
                        t[j]=x_grid[k]
                        break
                    if t_ratio[j,k+1]<c:
                        t[j]=x_grid[k]
                        break
        # calculate the FDR
        num_dis=0 
        num_fd =0 
        for i in range(num_case):
            num_dis+=np.sum(p[X==i] < t[i])
            num_fd+=np.sum(X==i)*t[i]*fit_param[i,0]

        if np.divide(num_fd,np.float(np.amax([num_dis,1])))<alpha:
            t_opt=t
        else:
            break
    return t_opt

def BH(x, alpha = 0.05, n = None):
    x_s = sorted(x)
    if n is None:
        n = len(x_s)
    ic = 0
    for i in range(len(x_s)):
        if x_s[i] < i*alpha/float(n):
            ic = i
    return ic, x_s[ic]



def p_value_beta_fit(p, lamb=0.8, bin_num=50, vis=0):
    pi_0=np.divide(np.sum(p>lamb), p.shape[0] * (1-lamb))
    temp_p=np.zeros([0])
    step_size=np.divide(1,np.float(bin_num))
    fil_num=np.int(np.divide(pi_0*p.shape[0],bin_num))+1
    for i in range(bin_num):
        p1=p[p>step_size*(i-1)]
        p1=p1[p1 <= step_size*i]
        choice_num= np.max(p1.shape[0] - fil_num,0)
        if choice_num > 1:
            choice=np.random.choice(p1.shape[0], choice_num)
            temp_p=np.concatenate([temp_p,p1[choice]]).T
    if vis==1:
        plt.figure()
        plt.hist(temp_p, bins=100, normed=True)       
    a, b, loc, scale = beta.fit(temp_p,floc=0,fscale=1)
    return pi_0, a, b
def beta_mixture_pdf(x,pi_0,a,b):
    return beta.pdf(x,a,b)*(1-pi_0)+pi_0



def Opt_t_cal_discrete(p, X, num_case=2,step_size=0.0001,ub=0.05,n_samples=2000,alpha=0.05):
    # Fit the beta mixture parameters
    fit_param=np.zeros([num_case, 3])
    for i in range(num_case):
        fit_param[i,:]=p_value_beta_fit(p[X==i])

    # Calculating the ratios 
    t_opt=np.zeros([num_case])
    max_idx=np.argmin(fit_param[:,0])
    x_grid = np.arange(0, ub, step_size)
    t_ratio=np.zeros([num_case,x_grid.shape[0]])
    for i in range(num_case):
        t_ratio[i,:] = np.divide(beta_mixture_pdf(
            x_grid,fit_param[i,0],fit_param[i,1],fit_param[i,2]), fit_param[i,0])

    # Increase the threshold
    for i in range(len(x_grid)):
        t=np.zeros([num_case])
        # undate the search optimal threshold
        t[max_idx]=x_grid[i]
        c=t_ratio[max_idx,i]
        for j in range(num_case):
            if j != max_idx: 
                for k in range(len(x_grid)):
                    if k == len(x_grid)-1:
                        t[j]=x_grid[k]
                        break
                    if t_ratio[j,k+1]<c:
                        t[j]=x_grid[k]
                        break
        # calculate the FDR
        num_dis=0 
        num_fd =0 
        for i in range(num_case):
            num_dis+=np.sum(p[X==i] < t[i])
            num_fd+=np.sum(X==i)*t[i]*fit_param[i,0]

        if np.divide(num_fd,np.float(np.amax([num_dis,1])))<alpha:
            t_opt=t
        else:
            break
    print('inside Opt_t_cal_discrete')

    return t_opt

def result_summary(h,pred):
    print("Num of alternatives:",np.sum(h))
    print("Num of discovery:",np.sum(pred))
    print("Num of true discovery:",np.sum(pred * h))
    print("Actual FDR:", 1-np.sum(pred * h) / np.sum(pred))
    
def softmax_prob_cal(X,Centorid, intensity=1):
    dist=np.zeros([n_samples,num_clusters])
    dist+=np.sum(X*X,axis=1, keepdims=True)
    dist+=np.sum(centroid.T*centroid.T,axis=0, keepdims=True)
    dist -= 2*X.dot(centroid.T)
    dist=np.exp(dist*intensity)
    dist /= np.sum(dist,axis=1, keepdims=True)

    return dist


def get_network(num_layers = 10, node_size = 10, dim = 1, scale = 1, cuda = False):
    
    
    class Model(nn.Module):
        def __init__(self, num_layers, node_size, dim):
            super(Model, self).__init__()
            l = []
            l.append(nn.Linear(dim,node_size))
            l.append(nn.LeakyReLU(0.1))
            for i in range(num_layers - 2):
                l.append(nn.Linear(node_size,node_size))
                l.append(nn.LeakyReLU(0.1))

            l.append(nn.Linear(node_size,1))
            l.append(nn.Sigmoid())

            self.layers = nn.Sequential(*l)

        def forward(self, x):
            x = self.layers(x)
            x = 0.5 * scale * x 
            return x
    
    network = Model(num_layers, node_size, dim)
    print('inside get_network')

    if cuda:
        return network.cuda()
    else:
        return network


def train_network_to_target_p(network, optimizer, x, target_p, num_it = 50, dim=3, cuda=False, batch_size=32):
    print('inside train_network_to_target_p')

    l1loss = nn.L1Loss()
    n_samples = len(x)
    loss_hist = []

    if cuda:
        network = network.cuda()
    for iteration in range(num_it):
        loss_batch = 0

        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()

            x_batch = x[i:i+batch_size]
            x_input = Variable(torch.from_numpy(x_batch.astype(np.float32).reshape(-1, dim)))
            target_batch = target_p[i:i+batch_size]
            target = Variable(torch.from_numpy(target_batch.astype(np.float32)))
            if cuda:
                x_input = x_input.cuda()
                target = target.cuda()

            output = network.forward(x_input)
            loss = l1loss(output, target)
            loss.backward()
            loss_batch += loss.data
        optimizer.step()
        loss_hist.append(loss_batch)
    print('almost outside train_network_to_target_p')
    return loss_hist

def train_network(network, optimizer, x, p, num_it = 50, alpha = 0.05, dim = 3, lambda_ = 20, lambda2_ = 1e3, cuda = False, fdr_scale = 1, mirror = 1, batch_size=32):
    n_samples = len(x)
    loss_hist = []
    soft_compare = nn.Sigmoid()
    relu = nn.ReLU()

    for iteration in range(num_it):
        loss_batch = 0
        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()

            x_batch = x[i:i+batch_size]
            x_input = Variable(torch.from_numpy(x_batch.astype(np.float32).reshape(-1, dim)))
            target_batch = p[i:i+batch_size]
            target = Variable(torch.from_numpy(target_batch.astype(np.float32)))
            if cuda:
                x_input = x_input.cuda()
                target = target.cuda()

            output = network.forward(x_input)
            s = torch.sum(soft_compare((output - target) * lambda2_)) / batch_size
            s2 = torch.sum(soft_compare((target - (mirror - output * fdr_scale)) * lambda2_)) / batch_size /float(fdr_scale)#false discoverate rate(over all samples)
            gain = s  - lambda_ * relu((s2 - s * alpha)) 

            loss = -gain
            loss.backward()
            loss_batch += loss.data
    
        optimizer.step()
        loss_hist.append(loss_batch)
    print('inside train_network')

    return loss_hist, s, s2

def opt_threshold(x, p, k, intensity = 1):
    n_samples = x.shape[0]
    
    if len(x.shape) == 1:
        x = np.expand_dims(x,1)
    km = KMeans(n_clusters = k)
    cluster = km.fit_predict(x)
    opt = Opt_t_cal_discrete(p, cluster, num_case = k, step_size=0.00001)
    #p_target = opt[cluster]
    
    x2 = x.repeat(k, axis = 1)
    center = km.cluster_centers_.repeat(n_samples, axis = 1).T

    e = np.exp (- (x2 - center) ** 2 / intensity)
    s = np.expand_dims(np.sum(e, axis = 1),1)
    prob = e/s
    opt = Opt_t_cal_discrete(p, cluster, num_case = 10, step_size=0.00001)
    p_target = prob.dot(opt)
    print('inside opt_threshold')
    return p_target


def opt_threshold_multi(x, p, k, intensity = 1, alpha = 0.05):
    n_samples = x.shape[0]

    km = KMeans(n_clusters = k)
    cluster = km.fit_predict(x)
    opt = Opt_t_cal_discrete(p, cluster, num_case = k, step_size=0.00001, alpha = alpha)
    center = np.expand_dims(km.cluster_centers_, axis = -1).repeat(n_samples, axis = -1).T
    x2 = np.expand_dims(x, axis = -1).repeat(k, axis = -1)

    dist = x2 - center
    dist = np.sum((x2 - center) ** 2, axis = 1)


    e = np.exp (- dist / intensity)
    s = np.expand_dims(np.sum(e, axis = 1),1)
    prob = e/s
    p_target = prob.dot(opt)
    print('inside opt_threshold_multi')

    return p_target

def get_scale(network, x, p, dim = 3, cuda = False, lambda_ = 20, lambda2_ = 1e3, alpha = 0.05, fit = False, scale = 1, fdr_scale = 1, mirror  = 1, batch_size = 32):
    n_samples = len(x)
    loss_hist = []
    soft_compare = nn.Sigmoid()
    relu = nn.ReLU()
    
    hi = 10.0
    low = 0.1
    current = scale
    total_s = 0
    total_s2 = 0
    if fit:
        for i in range(100):
            for i in range(0, n_samples, batch_size):
                x_batch = x[i:i+batch_size]
                x_input = Variable(torch.from_numpy(x_batch.astype(np.float32).reshape(-1, dim)))

                target_batch = p[i:i+batch_size]
                target = Variable(torch.from_numpy(target_batch.astype(np.float32)))

                if cuda:
                    x_input = x_input.cuda()
                    target = target.cuda()

                output = network.forward(x_input) * current

                # Compute soft values for the batch
                soft_output = soft_compare((output - target) * lambda2_)
                soft_mirror = soft_compare(target - (mirror - output * fdr_scale) * lambda2_)

                s = torch.sum(soft_output) / batch_size  # Discount rate for the batch
                s2 = torch.sum(soft_mirror) / (batch_size * fdr_scale)  # False discovery rate for the batch

                total_s += s
                total_s2 += s2

            if (total_s2 / total_s).cpu().data > alpha:
                hi = current
                current = (low + current) / 2
            else:
                low = current
                current = (hi + current) / 2
        
    return current, (total_s2 / total_s).cpu().data
