#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=weak_2.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 2 ../main 311 parallel
