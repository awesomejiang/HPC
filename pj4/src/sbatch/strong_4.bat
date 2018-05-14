#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=strong_4.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 4 ../main 600 parallel