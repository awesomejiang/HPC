#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=chunk_1.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 32 ../main dynamic 1