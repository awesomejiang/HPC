# HPC Project 6: Nbody Problem

## Compilation  
    cd src/
    make

P.S. You can switch between serial/openmp/mpi versions by modifying head of `makefile`       

## Running
    mpirun -n rank ./main [n_bodies] [n_iterations] [output filename]

### Plotting
    python plot.py

1. Plotting script depends on python library `matplotlib`
2. It will find `nbody.dat` under same directory and plot based on it.