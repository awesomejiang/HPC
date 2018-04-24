#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

module load openmpi/2.0.1+gcc-6.1

mpirun -n 9 ../../main 2400 400 1.0 1.0e3 5.0e-7 2.85e-7 1 mpi_non_blocking silence