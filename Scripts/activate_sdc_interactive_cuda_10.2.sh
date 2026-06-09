#!/usr/bin/env bash
conpath=/mnt/appsource/local/CAS/software/anaconda3/2021.5
module_bin=/mnt/appsource/modules/current/init/bash 
# modules_dir=/home/barthelemyx/modulefiles_2
modules_dir=/mnt/appsource/local/CAS/software/anaconda3/Environments/deep_learning_tensorflow_gpu_cuda_10.2/init_scripts/modules
#source /mnt/scratch/narclim/modulefiles/cas_env.sh

source ${module_bin}
# module use ${modules_dir}/SLES15
module use ${modules_dir}

module load anaconda3/2021.5
source ${conpath}/init.sh
conda activate /mnt/appsource/local/CAS/software/anaconda3/Environments/deep_learning_tensorflow_gpu_cuda_10.2
module load slurm
srun -N1 -w sdccomp07 -p GPU --pty bash