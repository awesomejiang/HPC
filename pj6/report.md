MPI ONLY: 

    mpirun -n 128 ../main -n 102400 -i 10 -o mpi.dat -t 1
13.44 s

Hybrid:

    mpirun -n 8 ../main -n 102400 -i 10 -o openmp.dat -t 16

28.74 s