#!/bin/bash
#
#SBATCH --job-name=DeepFDR
#SBATCH --nodes=1
#SBATCH --mem=2500GB
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00

module load python/intel/3.8.6

cd /scratch/tk2737/DeepFDR/ADNI

python3.8 gem.py --seed 0 --mode single_label --x_path /scratch/tk2737/ADNI --savepath /scratch/tk2737/ADNI --num_cpus 25
