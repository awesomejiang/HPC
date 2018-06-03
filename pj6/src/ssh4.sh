#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --output=ssh4.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 4 ./main -n 102400 -i 10 -t 16