#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --exclusive

module load mvapich2

mpirun -n 4 ../../main 10000 400 1.0 1.0e3 5.0e-7 2.85e-7 4 hybrid silence