import os
import argparse
import numpy as np
import time
import random 
import torch

class Model3:
    def __init__(self, args):
        self.args = args

        self.datapath = '../data/model3/'
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        self.label = torch.from_numpy(np.load(self.datapath + 'label/label.npy').astype('float32')).cuda()
        self.cov_matrix_path = './cov_matrix.npy'
        self.x_path = self.datapath + 'x_gpu.npy'

        if not os.path.exists(self.cov_matrix_path):
            self.precompute_cov_matrix()
            self.cov = torch.from_numpy(np.load(self.cov_matrix_path).astype('float32')).cuda()
        else:
            self.cov = torch.from_numpy(np.load(self.cov_matrix_path).astype('float32')).cuda()

    def precompute_cov_matrix(self):
        self.map = torch.zeros((self.args.dim,self.args.dim,self.args.dim,3), dtype=torch.float32)
        for i in range(self.args.dim):
            for j in range(self.args.dim):
                for k in range(self.args.dim):
                    self.map[i][j][k][0] = i
                    self.map[i][j][k][1] = j
                    self.map[i][j][k][2] = k

        self.map = torch.reshape(self.map, (self.args.dim**3,-1))        
        self.cov = torch.pow(0.5, torch.linalg.norm(self.map[:, None, :] - self.map[None, :, :], axis=-1))
        torch.save(self.cov, self.cov_matrix_path)

    def short_range_x(self):
        m1 = self.args.mu * self.label.reshape(self.args.dim**3,).cuda()
        d1 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m1, covariance_matrix=self.cov)
        m0 = torch.zeros(self.args.dim**3).cuda()
        d0 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m0, covariance_matrix=self.cov)

        v0 = d0.sample((self.args.subject,)).cuda()
        v1 = d1.sample((self.args.subject,)).cuda()

        x = ((1.0/self.args.subject)*torch.sum(v0, axis=0) - (1.0/self.args.subject)*torch.sum(v1, axis=0)) / (torch.sqrt(torch.tensor([(1.0/self.args.subject)+(1.0/self.args.subject)]))).cuda()
        print(x)
        x = x.reshape(self.args.dim,self.args.dim,self.args.dim)

        np.save(self.x_path, x.cpu().detach().numpy())
        return

    def long_range_x(self):
        return 

    def graph_x(self):
        return

def main():
    os.environ["OMP_NUM_THREADS"] = "1"  
    os.environ["MKL_NUM_THREADS"] = "1"  
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Model3')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--subject', default=200, type=int)
    parser.add_argument('--mu', default=1, type=float)
    parser.add_argument('--algo', default='short', type=str)
    parser.add_argument('--dim', default=30, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    m3 = Model3(args)
    start = time.time()
    if args.algo == 'short':
        m3.short_range_x()
    elif args.algo == 'long':
        m3.long_range_x()
    elif args.algo == 'graph':
        m3.graph_x()

    print("Elapsed: ", time.time()-start)

if __name__ == '__main__':
    main()
