#! /bin/bash

perf="strong weak"
num="1 2 4 8 16 32 64"

module load openmpi/2.0.1+gcc-6.1

cd src

make

cd sbatch

for p in $perf
do
	for n in $num
	do
		sbatch $p\_$n.bat
	done
done

cd ..

make clean