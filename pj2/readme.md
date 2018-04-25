# HPC project1
Implementation of 2d-advection simulation(Distributed memory version).

## main code(C++)
Source code in src/main and src/include/

### Compile
    make
gcc/6 or newer and mpic++ needed.

### Run
        ./main [N] [NT] [L] [T] [u] [v] [thread] [mode] [silence](optional)
1. gaussian spread args(sigma_x and sigma_y) is hard-coded in driver. sigma_x = sigma_y = L/4       
2. mode: serial, threads, mpi_blocking, mpi_non_blocking, hybrid
3. thread must be set to 1 under non-mpi modes.
4. Result matrice will be ouput into "[mode].out" as binary file.       

## Plot
Source code in src/plot.py      
Plot script for data.

### Run
    Show animation:     
        python ./plot.py example.out
    Save animation:     
        python ./plot.py example.out save
1. library "numpy" and "matplotlib" needed for running script.      
2. Animation file will be saved as "example.gif". External software "imagemagick" needed for saving gif.