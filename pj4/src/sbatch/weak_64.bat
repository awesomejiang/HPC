#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=weak_64.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 64 ../main 1760 parallel
