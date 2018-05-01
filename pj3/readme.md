# HPC project3
## Part1
Implementation of ping-pong latency and bandwidth test program.

### main code(C++)
Source code in src/main.        

#### Compile
    make
gcc/6 or newer and mpic++ needed.

#### Run

##### latency
        mpirun -n 2 ./latency [N]
1. mpi rank can only be 2.    
2. N should be big enough(>10000) to obtain accurate latency.      

##### bandwidth
        mpirun -n 2 ./bandwidth [UNIT]
1. mpi rank can only be 2.    
2. UNIT should be `KB/MB/GB`.

## Part2
Implementation of julia set(serial/mpi-staitc/mpi-dynamic).

### main code(C++)
Source code in src/main and src/include.        

#### Compile
    make
gcc/6 or newer and mpic++ needed.

#### Run
        mpirun -n 2 ./latency [N]
1. mpi rank can only be 2.    
2. N should be big enough(>10000) to obtain accurate latency.      

### Plot
Source code in `src/plot.py`        
Plot script for data.

#### Run
    Show fig:     
        python ./plot.py example.out
    Save fig:     
        python ./plot.py example.out save
library "numpy" and "matplotlib" needed for running script.     