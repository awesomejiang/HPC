#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=inter_latency.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 2 ../latency 100000