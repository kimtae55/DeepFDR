import numpy as np
import os
import math
import time
import sys
import pandas as pd
import argparse
import textwrap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test_Statistics Generation')
    parser.add_argument('--seed', default = 0, type=int)
    parser.add_argument('--mode', type=str, help='Supported modes: gem_gen, gt_gen, label_gen, data_gen', required=True)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--savepath', type=str)
    parser.add_argument('--simulation_path', default="./", type=str)
    parser.add_argument('--limit', default=600, type=int)
    parser.add_argument('--p_0', default=0.5, type=float)
    parser.add_argument('--p_1', default=0.5, type=float)
    parser.add_argument('--mu_0', default=0.5, type=float)
    parser.add_argument('--mu_1', default=0.5, type=float)
    parser.add_argument('--sig_0', default=0.5, type=float)
    parser.add_argument('--sig_1', default=0.5, type=float)
    args = parser.parse_args()
    rng_seed = args.seed
    start = args.start # 0, 30, 60, 90, 120, 150, 180, 210, 240, 270
    limit = args.limit
    mode = args.mode

    if mode == "gt_gen":
        filepath = os.path.join(args.simulation_path, str(rng_seed), "groundtruth")
        savepath = '/scratch/tk2737/DeepFDR/data/sim/'
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        with open(os.path.join(filepath, 'run_gt.sbatch'), 'w') as outfile:
            script = textwrap.dedent(f'''
            #!/bin/bash
            #
            #SBATCH --job-name=GTG
            #SBATCH --nodes=1
            #SBATCH --mem=32GB
            #SBATCH --cpus-per-task=20
            #SBATCH --time=24:00:00

            function cleanup_tmp_dir()
            {{
                if [[ "$TMPDIR" != "" ]] && [[ -d $TMPDIR ]]; then
                    rm -rf $TMPDIR
                fi
            }}

            trap cleanup_tmp_dir SIGKILL EXIT

            export TMPDIR=/vast/tk2737/tmp/tmp-job-$SLURM_JOB_ID
            mkdir -p $TMPDIR

            module purge
            export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

            singularity exec --nv --overlay /scratch/tk2737/singularity/deepFDR/overlay-50G-10M.ext3:rw \\
            /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; \\
            cd /scratch/tk2737/DeepFDR/ADNI; python gt.py --datapath {savepath} --seed {rng_seed}"        
            ''')
            outfile.write(script)
    elif mode == "data_gen":
        #python train_data.py --seed 0 --p_0 0.5 --p_1 0.5 --mu_0 -3.0 --mu_1 2.0 --sig_0 1.0 --sig_1 1.0 --label_path ../data/wnet/direct_x/mu_n3_2/label/label.txt --out_path ../data/wnet/direct_x/mu_n3_2
        savepath_list = []
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_8_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_4_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_2_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_1_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_5_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_25_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_125_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n1_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n15_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n2_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n25_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n3_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n35_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n4_2/')

        params_list = []
        params_list.append([0.5, 0.5, -2.0, 2.0, 8.0, 1.0])
        params_list.append([0.5, 0.5, -2.0, 2.0, 4.0, 1.0])
        params_list.append([0.5, 0.5, -2.0, 2.0, 2.0, 1.0])
        params_list.append([0.5, 0.5, -2.0, 2.0, 1.0, 1.0])
        params_list.append([0.5, 0.5, -2.0, 2.0, 0.5, 1.0])
        params_list.append([0.5, 0.5, -2.0, 2.0, 0.25, 1.0])
        params_list.append([0.5, 0.5, -2.0, 2.0, 0.125, 1.0])
        params_list.append([0.5, 0.5, -1.0, 2.0, 1.0, 1.0])
        params_list.append([0.5, 0.5, -1.5, 2.0, 1.0, 1.0])
        params_list.append([0.5, 0.5, -2.0, 2.0, 1.0, 1.0])
        params_list.append([0.5, 0.5, -2.5, 2.0, 1.0, 1.0])
        params_list.append([0.5, 0.5, -3.0, 2.0, 1.0, 1.0])
        params_list.append([0.5, 0.5, -3.5, 2.0, 1.0, 1.0])
        params_list.append([0.5, 0.5, -4.0, 2.0, 1.0, 1.0])

        for i, path in enumerate(savepath_list):
            labelpath = '/scratch/tk2737/DeepFDR/data/sim/label.npy'
            savepath = path
            params = params_list[i]
            filepath = os.path.join(args.simulation_path, str(rng_seed), "data", os.path.basename(os.path.normpath(savepath)))
            if not os.path.isdir(filepath):
                os.makedirs(filepath)

            with open(os.path.join(filepath, 'run_dg' + str(rng_seed) + '.sbatch'), 'w') as outfile:
                script = textwrap.dedent(f'''
                #!/bin/bash
                #
                #SBATCH --job-name=GTG
                #SBATCH --nodes=1
                #SBATCH --mem=34GB
                #SBATCH --cpus-per-task=20
                #SBATCH --time=24:00:00

                function cleanup_tmp_dir()
                {{
                    if [[ "$TMPDIR" != "" ]] && [[ -d $TMPDIR ]]; then
                        rm -rf $TMPDIR
                    fi
                }}

                trap cleanup_tmp_dir SIGKILL EXIT

                export TMPDIR=/vast/tk2737/tmp/tmp-job-$SLURM_JOB_ID
                mkdir -p $TMPDIR

                module purge
                export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

                singularity exec --nv --overlay /scratch/tk2737/singularity/deepFDR/overlay-50G-10M.ext3:rw \\
                /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; \\
                cd /scratch/tk2737/DeepFDR/ADNI; python test_statistic.py --savepath {savepath} \\
                --seed {rng_seed} --pL {params[0]} {params[1]} --muL {params[2]} {params[3]} --sigmaL2 {params[4]} {params[5]} --distribution gaussian_bootstrap \\
                --sample_size 6000 --labelpath {labelpath}"        
                ''')
                outfile.write(script)
    elif mode == "gem_gen":
        savepath_list = []
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_8_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_4_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_2_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_1_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_5_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_25_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/sigma_125_1/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n1_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n15_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n2_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n25_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n3_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n35_2/')
        savepath_list.append('/scratch/tk2737/DeepFDR/data/sim/' + str(rng_seed) + '/mu_n4_2/')
        
        for path in savepath_list:
            savepath = path
            x_path = os.path.join(savepath, 'data', 'data.npz')
            labelpath = '/scratch/tk2737/DeepFDR/data/sim/label.npy'
            filepath = os.path.join(args.simulation_path, str(rng_seed), "gem", os.path.basename(os.path.normpath(savepath)))
            
            if not os.path.isdir(filepath):
                os.makedirs(filepath)
                
            script = textwrap.dedent(f'''
            #!/bin/bash
            #
            #SBATCH --job-name=GTG
            #SBATCH --nodes=1
            #SBATCH --mem=64GB
            #SBATCH --cpus-per-task=20
            #SBATCH --time=48:00:00

            function cleanup_tmp_dir()
            {{
                if [[ "$TMPDIR" != "" ]] && [[ -d $TMPDIR ]]; then
                    rm -rf $TMPDIR
                fi
            }}

            trap cleanup_tmp_dir SIGKILL EXIT

            export TMPDIR=/vast/tk2737/tmp/tmp-job-$SLURM_JOB_ID
            mkdir -p $TMPDIR

            module purge
            export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

            singularity exec --nv --overlay /scratch/tk2737/singularity/deepFDR/overlay-50G-10M.ext3:rw \\
            /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; cd \\
            /scratch/tk2737/DeepFDR/ADNI; python gem.py --seed {rng_seed} --mode {mode} --savepath {savepath} --x_path \\
            {x_path} --labelpath {labelpath} --num_cpus 4"
                    ''')
            with open(os.path.join(filepath, 'run_gem.sbatch'), 'w') as outfile:
                outfile.write(script)










