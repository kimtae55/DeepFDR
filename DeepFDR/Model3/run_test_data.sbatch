#!/bin/bash
#
#SBATCH --job-name=DeepFDR
#SBATCH --nodes=1
#SBATCH --mem=36GB
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00

module load python/intel/3.8.6

cd /scratch/tk2737/DeepFDR;
source venv/bin/activate;
cd /scratch/tk2737/DeepFDR/Model3

python test_data.py --algo short --mu 1.0