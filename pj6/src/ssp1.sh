#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=ssp1.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 16 ./main -n 102400 -i 10 -t 1