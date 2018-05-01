#! /bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=chunk.out

module load openmpi/2.0.1+gcc-6.1

mpirun -n 32 ../main dynamic 10
mpirun -n 32 ../main dynamic 20
mpirun -n 32 ../main dynamic 30
mpirun -n 32 ../main dynamic 40
mpirun -n 32 ../main dynamic 50
mpirun -n 32 ../main dynamic 60
mpirun -n 32 ../main dynamic 70
mpirun -n 32 ../main dynamic 80
mpirun -n 32 ../main dynamic 90
mpirun -n 32 ../main dynamic 100
mpirun -n 32 ../main dynamic 150
mpirun -n 32 ../main dynamic 200
mpirun -n 32 ../main dynamic 400
mpirun -n 32 ../main dynamic 500
