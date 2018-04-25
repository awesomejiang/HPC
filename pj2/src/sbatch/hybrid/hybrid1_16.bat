#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --output=hybrid1_16.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 1 ../../main 10000 400 1.0 1.0e3 5.0e-7 2.85e-7 16 hybrid silence