# HPC Project 4: Distributed Conjugate Gradient Method

## Compilation
    cd src/
    make

## Running
### main program
    mpirun -n [rank] ./main [size] [mode]
1. mode could be `serial` or `parallel`.    
2. While running serial code, make sure rank is set to 1. Or, you can directly run withour `mpirun` command.

### plotting script
    python plot.py [filename.out] [save](optional)

1. Plotting script depends on python library `matplotlib`
2. Argument `save` is optional. If added, figure will save in a file named `filename.png`