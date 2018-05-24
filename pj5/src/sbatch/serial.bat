#! /bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=serial_sbatch.out

module load cuda/8.0
module load gcc/4.8

../main serial 1000 1000
../main serial 1000 10000
../main serial 1000 100000
../main serial 1000 1000000
../main serial 1000 10000000
../main serial 1000 100000000
../main serial 1000 1000000000