# 1. calculate p value from z-test statistics
# 2. run BH-method from p value
# 3. run Q-value method from p value
# generate gamma_file using the output probability
# calculate fdr, fnr, atp

import numpy as np
import sys
import os
from matplotlib import pyplot as plt

replication_list = ['0']
mu_list = ['mu_n4_2', 'mu_n35_2', 'mu_n3_2', 'mu_n25_2', 'mu_n2_2', 'mu_n15_2', 'mu_n1_2']
sigma_list = ['sigma_125_1', 'sigma_25_1', 'sigma_5_1', 'sigma_1_1', 'sigma_2_1', 'sigma_4_1', 'sigma_8_1']

root = '/scratch/tk2737/DeepFDR/data/sim/'

for rep in replication_list:
    for mu in mu_list:
        path = root + rep + '/' + mu 
        label = root + 'label.npy'

    for sig in sigma_list:
        path = root + rep + '/' + sig 
        label = root + 'label.npy'

model2 = {'fdr':[[] for _ in range(len(mu_list))], 
          'fnr':[[] for _ in range(len(mu_list))], 
          'atp':[[] for _ in range(len(mu_list))]
          }
qval = {'fdr':[[] for _ in range(len(mu_list))], 
          'fnr':[[] for _ in range(len(mu_list))], 
          'atp':[[] for _ in range(len(mu_list))]
          }
bh = {'fdr':[[] for _ in range(len(mu_list))], 
          'fnr':[[] for _ in range(len(mu_list))], 
          'atp':[[] for _ in range(len(mu_list))]
          }
smooth = {'fdr':[[] for _ in range(len(mu_list))], 
          'fnr':[[] for _ in range(len(mu_list))], 
          'atp':[[] for _ in range(len(mu_list))]
          }

for i in replication_list:
    for j, mu_type in enumerate(mu_list):
        path = os.path.join(root, i, mu_type)

        with open(os.path.join(path, 'result/bh.txt'), 'r') as file:
            lines = file.readlines()
            bh['fdr'][j].append(float(lines[0].strip('fdr: ')))
            bh['fnr'][j].append(float(lines[1].strip('fnr: ')))
            bh['atp'][j].append(float(lines[2].strip('atp: ')))
           
        with open(os.path.join(path, 'result/qval.txt'), 'r') as file:
            lines = file.readlines()
            qval['fdr'][j].append(float(lines[0].strip('fdr: ')))
            qval['fnr'][j].append(float(lines[1].strip('fnr: ')))
            qval['atp'][j].append(float(lines[2].strip('atp: ')))   

        with open(os.path.join(path, 'result/stats.txt'), 'r') as file:
            lines = file.readlines()
            model2['fdr'][j].append(float(lines[1].strip('fdr: ')))
            model2['fnr'][j].append(float(lines[2].strip('fnr: ')))
            model2['atp'][j].append(float(lines[3].strip('atp: ')))  

        with open(os.path.join(path, 'result/smooth.txt'), 'r') as file:
            lines = file.readlines()
            smooth['fdr'][j].append(float(lines[1].strip('fdr: ')))
            smooth['fnr'][j].append(float(lines[2].strip('fnr: ')))
            smooth['atp'][j].append(float(lines[3].strip('atp: ')))  

with open(os.path.join(root, 'final_mu.txt'), 'w') as outfile:
    outfile.write('model2: \n')
    for key,val in model2.items():
        outfile.write(key + '\n')
        for i in range(len(val)):
            outfile.write(str(val[i]) + '\n')
            outfile.write(str(np.mean(val[i])) + " " + str(np.std(val[i]))+ '\n')
    outfile.write('\nqval: \n')
    for key,val in qval.items():
        outfile.write(key + '\n')
        for i in range(len(val)):
            outfile.write(str(val[i])+ '\n')
            outfile.write(str(np.mean(val[i])) + " " + str(np.std(val[i]))+ '\n')
    outfile.write('\nbh: \n')
    for key,val in bh.items():
        outfile.write(key + '\n')
        for i in range(len(val)):
            outfile.write(str(val[i])+ '\n')
            outfile.write(str(np.mean(val[i])) + " " + str(np.std(val[i]))+ '\n')
    outfile.write('\nsmooth: \n')
    for key,val in smooth.items():
        outfile.write(key + '\n')
        for i in range(len(val)):
            outfile.write(str(val[i])+ '\n')
            outfile.write(str(np.mean(val[i])) + " " + str(np.std(val[i]))+ '\n')


model2 = {'fdr':[[] for _ in range(len(sigma_list))], 
          'fnr':[[] for _ in range(len(sigma_list))], 
          'atp':[[] for _ in range(len(sigma_list))]
          }
qval = {'fdr':[[] for _ in range(len(sigma_list))], 
          'fnr':[[] for _ in range(len(sigma_list))], 
          'atp':[[] for _ in range(len(sigma_list))]
          }
bh = {'fdr':[[] for _ in range(len(sigma_list))], 
          'fnr':[[] for _ in range(len(sigma_list))], 
          'atp':[[] for _ in range(len(sigma_list))]
          }
smooth = {'fdr':[[] for _ in range(len(sigma_list))], 
          'fnr':[[] for _ in range(len(sigma_list))], 
          'atp':[[] for _ in range(len(sigma_list))]
          }

for i in replication_list:
    for j, sigma_type in enumerate(sigma_list):
        path = os.path.join(root, i, sigma_type)

        with open(os.path.join(path, 'result/bh.txt'), 'r') as file:
            lines = file.readlines()
            bh['fdr'][j].append(float(lines[0].strip('fdr: ')))
            bh['fnr'][j].append(float(lines[1].strip('fnr: ')))
            bh['atp'][j].append(float(lines[2].strip('atp: ')))
           
        with open(os.path.join(path, 'result/qval.txt'), 'r') as file:
            lines = file.readlines()
            qval['fdr'][j].append(float(lines[0].strip('fdr: ')))
            qval['fnr'][j].append(float(lines[1].strip('fnr: ')))
            qval['atp'][j].append(float(lines[2].strip('atp: ')))   

        with open(os.path.join(path, 'result/stats.txt'), 'r') as file:
            lines = file.readlines()
            model2['fdr'][j].append(float(lines[1].strip('fdr: ')))
            model2['fnr'][j].append(float(lines[2].strip('fnr: ')))
            model2['atp'][j].append(float(lines[3].strip('atp: ')))  

        with open(os.path.join(path, 'result/smooth.txt'), 'r') as file:
            lines = file.readlines()
            smooth['fdr'][j].append(float(lines[1].strip('fdr: ')))
            smooth['fnr'][j].append(float(lines[2].strip('fnr: ')))
            smooth['atp'][j].append(float(lines[3].strip('atp: ')))  

with open(os.path.join(root, 'final_sigma.txt'), 'w') as outfile:
    outfile.write('model2: \n')
    for key,val in model2.items():
        outfile.write(key + '\n')
        for i in range(len(val)):
            outfile.write(str(val[i]) + '\n')
            outfile.write(str(np.mean(val[i])) + " " + str(np.std(val[i]))+ '\n')
    outfile.write('\nqval: \n')
    for key,val in qval.items():
        outfile.write(key + '\n')
        for i in range(len(val)):
            outfile.write(str(val[i])+ '\n')
            outfile.write(str(np.mean(val[i])) + " " + str(np.std(val[i]))+ '\n')
    outfile.write('\nbh: \n')
    for key,val in bh.items():
        outfile.write(key + '\n')
        for i in range(len(val)):
            outfile.write(str(val[i])+ '\n')
            outfile.write(str(np.mean(val[i])) + " " + str(np.std(val[i]))+ '\n')
    outfile.write('\nsmooth: \n')
    for key,val in smooth.items():
        outfile.write(key + '\n')
        for i in range(len(val)):
            outfile.write(str(val[i])+ '\n')
            outfile.write(str(np.mean(val[i])) + " " + str(np.std(val[i]))+ '\n')

