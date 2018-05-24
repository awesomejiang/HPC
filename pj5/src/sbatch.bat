#! /bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=sbatch.out

module load cuda/8.0
module load gcc/4.8

./main cuda 1000 4 250
./main cuda 1000 40 250
./main cuda 1000 400 250
./main cuda 1000 4000 250
./main cuda 1000 40000 250
./main cuda 1000 400000 250
./main cuda 1000 4000000 250

./main serial 1000 1000
./main serial 1000 10000
./main serial 1000 100000
./main serial 1000 1000000
./main serial 1000 10000000
./main serial 1000 100000000
./main serial 1000 1000000000
