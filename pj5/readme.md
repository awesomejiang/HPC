# HPC Project 5: Cuda Ray Tracer

## Compilation  
    cd src/
    make

P.S. cuda/8.0(nvcc) or above is needed.   

## Running
### Serial version
    ./main serial [ndim] [rays]

### Cuda version
    ./main cuda [ndim] [blocks] [threads]
    
    (Note that #rays = blocks * threads)  

### Plotting
    python plot.py [filename.out] [save](optional)

1. Plotting script depends on python library `matplotlib`
2. Argument `save` is optional. If added, figure will save in a file named `filename.png`