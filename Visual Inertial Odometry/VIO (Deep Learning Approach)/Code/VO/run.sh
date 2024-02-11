#!/bin/bash
SBATCH -N 1
SBATCH -n 2
SBATCH --mem 500
SBATCH --gres=gpu:4
SBATCH --gpus-per-node=1
SBATCH -o Train.out
SBATCH --time=64:00:00
SBATCH --mail-user=usivaraman@wpi.edu
SBATCH --mail-type=ALL

# SBATCH -J test_cuda
SBATCH --output=slurm_outputs/cuda_test_out_%j.out
SBATCH --error=slurm_outputs/cuda_test_err_%j.err
srun -l python3 overall_code.py