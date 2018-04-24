import sys
import numpy as np

data1 = np.fromfile(sys.argv[1])
data2 = np.fromfile(sys.argv[2])

if(len(data1) != len(data2)):
	print "Size of 2 output files differ."

ctr = 0
for i in xrange(200*200*2):
	#if(data1[i] != data2[i]):
	if(abs(data1[i]-data2[i]) > 10e-10):
		ctr += 1
print ctr