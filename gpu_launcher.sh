#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --account=uo0780
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=log/%x.%j.out  # %x is the job name, %j is the job ID

source activate obta_paper
export FRIDOM_BACKEND="jax_gpu"

TOTAL_TASKS=$((SLURM_NNODES * 4))

srun -l --mpi=pmi2 python3 "$1"
