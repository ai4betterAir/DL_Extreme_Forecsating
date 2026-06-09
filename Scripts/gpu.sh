#!/usr/bin/env bash
conpath=/mnt/appsource/local/CAS/software/anaconda3/2022.10
module_bin=/mnt/appsource/modules/current/init/bash 
# modules_dir=/home/barthelemyx/modulefiles_2
modules_dir=/mnt/appsource/local/CAS/software/anaconda3/Environments/deep_learning_tensorflow_gpu_cuda_11.8/init_scripts/modules
#source /mnt/scratch/narclim/modulefiles/cas_env.sh

ulimit -s unlimited

source ${module_bin}
# module use ${modules_dir}/SLES15
module use ${modules_dir}

module load anaconda3/2022.10
source ${conpath}/init.sh
conda activate /mnt/appsource/local/CAS/software/anaconda3/Environments/deep_learning_tensorflow_gpu_cuda_11.8
module load slurm
module load cuda
srun -N1 -w sdccomp06 -p GPU --pty bash
