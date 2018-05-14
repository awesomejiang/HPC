# HPC Project 4
## Jiawei Jiang

## 1. Serial Experimentation
Run a simulation of 75*75 domain.   

### 1.1 Resources used
Mehtod | Memory(MB) | Time(sec)
-- | -- | --
Dense | 241.61 | 8.52
Sparse(OTF) | 0.21 | 0.02

### 1.2 Example plotting
![serial_fig](./fig/serial.png)

### 1.3 Estimations
Dense memory method will consume a memory of O(N*N), while OTF method only need O(N).(Approximately)    
If we scale the problem size ot 10000*10000:   
Dense memory will increase to  around 2*10^10 GB.   
Sparse memory will be 4 GB.

## 2. Parallel Experimentation

### 2.1 Example plotting
Here is the result of `mpirun -n 4 ./main 75 parallel`: 
![parallel_fig](./fig/parallel.png)