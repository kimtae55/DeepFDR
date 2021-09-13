#!/bin/bash
#
#SBATCH --job-name=GTG
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1

module load python3/intel/3.8.6
module load cuda/11.1.74
 
cd /scratch/tk2737/DeepFDR;
source venv/bin/activate;
cd /scratch/tk2737/DeepFDR/Model1/Dim3
python test_fdr.py 1
