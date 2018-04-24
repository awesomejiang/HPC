#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

module load mvapich2

mpirun -n 16 ../../main 3200 400 1.0 1.0e3 5.0e-7 2.85e-7 1 mpi_non_blocking silence