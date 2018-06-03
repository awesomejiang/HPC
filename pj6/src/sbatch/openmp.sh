#! /bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=sandyb
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --output=openmp.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 8 ../main -n 102400 -i 10 -o openmp.dat -t 16