#!/usr/bin/env bash
env_activate=/home/karalasa/sparse-lstm/bin/activate  # update this to the full path of your sparse-lstm env
module_bin=/mnt/appsource/modules/current/init/bash
#modules_dir=/mnt/appsource/local/CAS/software/anaconda3/2024.06/init_scripts/modules  # update if modules_dir path has changed

ulimit -s unlimited

source ${module_bin}
module use ${modules_dir}

module load slurm
#module load cuda/12.5.0_555.42.02
module load cuda/12.3.0_545.23.08

source ${env_activate}

export HDF5_USE_FILE_LOCKING=FALSE