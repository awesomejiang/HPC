# HPC project1
Implementation of 2d-advection simulation. Serial and parallel version.

## Serial
Source code in src/serial/      

### Compile
    make
gcc/6 or newer needed.

### Run
        ./serial [N] [NT] [L] [T] [u] [v]
1. gaussian spread args(sigma_x and sigma_y) is hard-coded in driver. sigma_x = sigma_y = L/4       
2. Result matrice will be ouput into "seiral.out"

## Parallel
Source code in src/parallel/      

### Compile
    make
gcc/6 or newer needed.

### Run
        ./serial [N] [NT] [L] [T] [u] [v] [threads]
1. gaussian spread args(sigma_x and sigma_y) is hard-coded in driver. sigma_x = sigma_y = L/4       
2. Result matrice will be ouput into "parallel.out"

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