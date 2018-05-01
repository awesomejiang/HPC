#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --output=static_32.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 32 ../main static