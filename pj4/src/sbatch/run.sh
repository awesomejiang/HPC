#! /bin/bash

perf="strong weak"
num="1 2 4 8 16 32 64"

for p in $perf
do
	for n in $num
	do
		sbatch $p\_$n.bat
	done
done
