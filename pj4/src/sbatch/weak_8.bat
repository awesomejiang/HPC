#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=weak_8.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 8 ../main 622 parallel
