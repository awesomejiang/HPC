#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=cuda_sbatch.out

module load cuda/8.0
module load gcc/4.8

../main cuda 1000 4 250
../main cuda 1000 40 250
../main cuda 1000 400 250
../main cuda 1000 4000 250
../main cuda 1000 40000 250
../main cuda 1000 400000 250
../main cuda 1000 4000000 250