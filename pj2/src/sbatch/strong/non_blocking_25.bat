#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=non_blocking_25.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 25 ../../main 4200 400 1.0 1.0e3 5.0e-7 2.85e-7 1 mpi_non_blocking silence