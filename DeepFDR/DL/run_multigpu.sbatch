#!/bin/bash
#
#SBATCH --job-name=DeepFDR
#SBATCH --nodes=1
#SBATCH --mem=36GB
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:4

module load python/intel/3.8.6
module load cuda/11.1.74

cd /scratch/tk2737/DeepFDR;
source venv/bin/activate;
cd /scratch/tk2737/DeepFDR/DL

python3.8 -m torch.distributed.launch --nproc_per_node=4 --master_port 6666 train.py 