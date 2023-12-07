from sideinfo_release import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import sys
import argparse
import os
import scipy.stats as stats


parser = argparse.ArgumentParser()
parser.add_argument('--mu', type=str, default = 'mu_n4_2')
parser.add_argument('--sigma', type=str, default = 'sigma_125_1')
parser.add_argument('--dim', type=int, default = 3,  help='dimension of data')
parser.add_argument('--init', type=int, default = 1,  help='number of inits')
parser.add_argument('--num_it', type=int, default = 1,  help='number of inits')
parser.add_argument('--out', type=str, default = 'test',  help='output_directory')
parser.add_argument('--prefix', type=str, default = 'http://localhost:8888/files',  help='url prefix')
parser.add_argument('--alpha', type=float, default = 0.1,  help='fdr')
parser.add_argument('--intensity', type=float, default = 1,  help='fdr')
parser.add_argument('--fdr_scale', type=float, default = 1,  help='fd scale')
parser.add_argument('--mirror', type=float, default = 1,  help='mirror')


opt = parser.parse_args()

opt.cuda = True

dim = opt.dim

# Define the dimensions of the cube
a, b, c = 30,30,30  # Replace with your desired dimensions

# Create coordinate arrays along each axis
x = np.arange(a)
y = np.arange(b)
z = np.arange(c)

# Create a grid of coordinates using NumPy's meshgrid
X, Y, Z = np.meshgrid(x, y, z)

# Reshape the coordinate grids into a (a*b*c, 3) array
x = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


root = '/scratch/tk2737/DeepFDR/data/sim'
#root = '/Users/taehyo/Dropbox/NYU/Research/Research/Code/deepfdr/DeepFDR/DL_unsupervised/data'
#root = 'C:\\Users\\taehy\\Dropbox\\NYU\\Research\\Research\\Code\\deepfdr\\DeepFDR\\DL_unsupervised\\data'

savepath_list = []
mu_list = [] 
sigma_list = []
mu_list.append(opt.mu)
sigma_list.append(opt.sigma)

for mu in mu_list:
    path = root + '/' + 'mu/' + mu
    savepath_list.append(path)

for sig in sigma_list:
    path = root + '/' + 'sigma/' + sig
    savepath_list.append(path)

for index in range(len(savepath_list)):
    fdrs = []
    fnrs = []
    atps = []
    times = []

    for rep in range(2):
        then = timeit.default_timer()

        data = np.ravel(np.load(os.path.join(savepath_list[index]+'/data0.2.npy'))[rep])
        p =  2.0*(1.0-stats.norm.cdf(np.fabs(data)))
        label = np.ravel(np.load(root+'/cubes0.2.npy')[0])
        h = label.copy()

        n_samples = len(x)

        grids = None
        x_prob = None

        if dim == 1:
            max_x = np.max(x)
            min_x = np.min(x)
            x_prob = np.arange(min_x, max_x, (max_x - min_x)/1000.0)
            x_prob = x_prob.reshape((len(x_prob), 1))
            x_prob = Variable(torch.from_numpy(x_prob.astype(np.float32)))

        elif dim == 2:
            max_x0 = np.max(x[:,0])
            min_x0 = np.min(x[:,0])
            max_x1 = np.max(x[:,1])
            min_x1 = np.min(x[:,1])
            x_prob0 = np.arange(min_x0, max_x0, (max_x0 - min_x0)/100.0)
            x_prob1 = np.arange(min_x1, max_x1, (max_x1 - min_x1)/100.0)
            X_grid, Y_grid = np.meshgrid(x_prob0, x_prob1)
            x_prob = Variable(torch.from_numpy(
            np.concatenate([[X_grid.flatten()], [Y_grid.flatten()]]).T.astype(np.float32)))
            grids = (X_grid, Y_grid)

        elif dim == 3:
            max_x0 = np.max(x[:, 0])
            min_x0 = np.min(x[:, 0])
            max_x1 = np.max(x[:, 1])
            min_x1 = np.min(x[:, 1])
            max_x2 = np.max(x[:, 2])
            min_x2 = np.min(x[:, 2])

            x_prob0 = np.arange(min_x0, max_x0, (max_x0 - min_x0) / 100.0)
            x_prob1 = np.arange(min_x1, max_x1, (max_x1 - min_x1) / 100.0)
            x_prob2 = np.arange(min_x2, max_x2, (max_x2 - min_x2) / 100.0)

            X_grid, Y_grid, Z_grid = np.meshgrid(x_prob0, x_prob1, x_prob2)

            x_prob = Variable(torch.from_numpy(
                np.concatenate([[X_grid.flatten()], [Y_grid.flatten()], [Z_grid.flatten()]]).T.astype(np.float32)))
            grids = (X_grid, Y_grid, Z_grid)


        if not x_prob is None:
            print(x_prob.size())
            if opt.cuda:
                x_prob = x_prob.cuda()

        #network = get_network(cuda = True, dim = dim)
        #optimizer = optim.Adagrad(network.parameters(), lr = 0.01)

        bhp = BH(p, alpha = opt.alpha)[1]
        lambda_param = 4/bhp
        print('lambda ', lambda_param)

        #from IPython import embed; embed()

        #select = np.logical_or(p < bhp * 10, p > 1 - bhp * 10)
        #x = x[select, :]
        #p = p[select]

        indices = np.random.permutation(x.shape[0])
        print(indices[:int(x.shape[0]/3)], indices[int(x.shape[0]/3) : int(x.shape[0]/3*2)], indices[int(x.shape[0]/3 * 2):])
        A = [indices[:int(x.shape[0]/3)], indices[int(x.shape[0]/3) : int(x.shape[0]/3*2)], indices[int(x.shape[0]/3 * 2):]]
        train = A
        val = [A[1], A[2], A[0]]
        test = [A[2], A[0], A[1]]
        outputs = []
        preds = []
        gts = []

        loss_hists1 = []
        loss_hists2 = []

        efdr = np.zeros((3,3))
        scales = np.zeros(3)

        ninit = opt.init



        if dim == 1:
            x = x.reshape((x.shape[0], 1))

        for i in range(3):
            networks = []
            scores = []
            loss_hist1_array = []
            loss_hist2_array = []
            for j in range(ninit):
                print("HI: ", i,j)
                network = get_network(num_layers = 10, cuda = opt.cuda, dim = dim, scale = opt.mirror)
                optimizer = optim.Adagrad(network.parameters(), lr = 0.01)
                train_idx = train[i]
                val_idx = val[i]
                test_idx = test[i]

                #network init
                try:
                    p_target = opt_threshold_multi(x[train_idx,:], p[train_idx], 10, alpha = opt.alpha)
                except:
                    p_target = np.ones(x[train_idx,:].shape[0]) * Storey_BH(p[train_idx], alpha = opt.alpha)[1]


                #plt.figure()
                #plt.scatter(x, p_target)
                loss_hist = train_network_to_target_p(network, optimizer, x[train_idx,:], p_target, num_it = opt.num_it, cuda= opt.cuda, dim = dim)
                loss_hist2, s, s2 = train_network(network, optimizer, x[train_idx,:], p[train_idx], num_it = opt.num_it, cuda = opt.cuda, dim = dim, alpha = opt.alpha, lambda2_ = lambda_param, fdr_scale = opt.fdr_scale)

                loss_hist_np = np.array([tensor.detach().cpu().numpy() for tensor in loss_hist2])
                score = np.mean(loss_hist_np[-100:])
                print(j,score)
                networks.append(network)
                scores.append(score)
                loss_hist1_array.append(loss_hist)
                loss_hist2_array.append(loss_hist2)

            idx = np.argmin(np.array(scores))

            loss_hist = loss_hist1_array[idx]
            loss_hist2 = loss_hist2_array[idx]
            network = networks[idx]


            scale, efdr[i,1] = get_scale(network, x[val_idx,:], p[val_idx], cuda = opt.cuda, lambda2_ = 5e12, fit = True, dim = dim, alpha = opt.alpha, fdr_scale = opt.fdr_scale, mirror = opt.mirror)

            scales[i] = scale
            if scale > 2 or scale < 0.5:
                print('Warning: abnormal scale factor, suggest rerun')

            n_samples = len(x[test_idx])
            x_input = Variable(torch.from_numpy(x[test_idx,:].astype(np.float32).reshape(n_samples ,dim)))
            p_input = Variable(torch.from_numpy(p[test_idx].astype(np.float32).reshape(n_samples ,1)))
            if opt.cuda:
                x_input = x_input.cuda()
                p_input = p_input.cuda()
            
            output = network.forward(x_input) * scale
            pred = (p_input < output).cpu().data.numpy()
            pred = pred[:,0].astype(np.float32)
            preds.append(pred)

            if not x_prob is None:
                outputs.append(network.forward(x_prob) * scale)

            gts.append(h[test_idx])


        preds = np.ravel(np.concatenate(preds))
        gts = np.ravel(np.concatenate(gts))

        TP = np.sum((gts == 1) & (preds == 1))
        TN = np.sum((gts == 0) & (preds == 0))
        FP = np.sum((gts == 0) & (preds == 1))
        FN = np.sum((gts == 1) & (preds == 0))

        atp = TP
        fdr = FP / np.sum(preds)
        fnr = FN / (preds.size - np.sum(preds))
        time = timeit.default_timer() - then

        print(atp, fdr, fnr, time)

        fdrs.append(fdr)
        fnrs.append(fnr)
        atps.append(atp)
        times.append(time)

    with open(os.path.join(savepath_list[index], str(opt.init) + '_' + str(opt.num_it) + '_neuralfdr_0.2.txt'), 'w') as outfile:
        mfdr, sfdr = np.mean(fdrs), np.std(fdrs)
        mfnr, sfnr = np.mean(fnrs), np.std(fnrs)
        matp, satp = np.mean(atps), np.std(atps)
        mtime, stime = np.mean(times), np.std(times)
        outfile.write(f'fdr: {mfdr} ({sfdr})\n')
        outfile.write(f'fnr: {mfnr} ({sfnr})\n')
        outfile.write(f'atp: {matp} ({satp})\n')

