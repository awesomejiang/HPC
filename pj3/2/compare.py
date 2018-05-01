import sys
import numpy as np
import math

data1 = np.fromfile(sys.argv[1], dtype='uint32')
data2 = np.fromfile(sys.argv[2], dtype='uint32')

if(len(data1) != len(data2)):
	print "Size of 2 output files differ."
	print len(data1), len(data2)

ctr = 0
for i in xrange(len(data1)):
	if(data1[i] != data2[i]):
		ctr += 1
print ctr