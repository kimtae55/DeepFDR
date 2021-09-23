import os
import argparse
import numpy as np
import time
import random 

class Model3:
    def __init__(self, args):
        self.args = args
        self.datapath = '../data/model3/'
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        self.label = np.load(self.datapath + 'label/label.npy')
        self.cov_matrix_path = './cov_matrix.npy'
        self.x_path = self.datapath + 'x.npy'
        self.cov = np.load(self.cov_matrix_path)
        if not os.path.exists(self.cov_matrix_path):
            self.precompute_cov_matrix()
            self.cov = np.load(self.cov_matrix_path)
        else:
            self.cov = np.load(self.cov_matrix_path)

    def precompute_cov_matrix(self):
        self.map = np.zeros((self.args.dim,self.args.dim,self.args.dim,3), dtype=np.float32)
        for i in range(self.args.dim):
            for j in range(self.args.dim):
                for k in range(self.args.dim):
                    self.map[i][j][k][0] = i
                    self.map[i][j][k][1] = j
                    self.map[i][j][k][2] = k

        self.map = np.reshape(self.map, (self.args.dim**3,-1))        
        self.cov = np.power(0.5, np.linalg.norm(self.map[:, None, :] - self.map[None, :, :], axis=-1))
        np.save(self.cov_matrix_path, self.cov)

    def short_range_x(self):
        m = self.args.mu * self.label.reshape(self.args.dim**3,)
        v1 = np.random.multivariate_normal(mean=m, cov=self.cov, size=self.args.subject)
        m = np.zeros(self.args.dim**3)
        v0 = np.random.multivariate_normal(mean=m, cov=self.cov, size=self.args.subject)

        x = ((1.0/self.args.subject)*np.sum(v0, axis=0) - (1.0/self.args.subject)*np.sum(v1, axis=0)) / (np.sqrt((1.0/self.args.subject)+(1.0/self.args.subject)))
        x = x.reshape(self.args.dim,self.args.dim,self.args.dim)

        np.save(self.x_path, x)
        return

    def long_range_x(self):
        return 

    def graph_x(self):
        return

def main():
    os.environ["OMP_NUM_THREADS"] = "1"  
    os.environ["MKL_NUM_THREADS"] = "1"  
    os.environ["NUMEXPR_NUM_THREADS"] = "1"   
       
    parser = argparse.ArgumentParser(description='Model3')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--subject', default=200, type=int)
    parser.add_argument('--mu', default=1, type=float)
    parser.add_argument('--algo', default='short', type=str)
    parser.add_argument('--dim', default=30, type=int)
    args = parser.parse_args()

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
