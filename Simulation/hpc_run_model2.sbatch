#!/bin/bash
#
#SBATCH --job-name=GTG
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00

module --ignore-cache load python3/intel/3.8.6
 
cd /scratch/tk2737/DeepFDR;
source venv/bin/activate;
cd /scratch/tk2737/DeepFDR/Model2
python test_theta1.py 150 && python test_data_cpu.py 150 -4.0 2.0 1.0 1.0 && python test_fdr2_direct.py 150 && python multi_testing.py 150 