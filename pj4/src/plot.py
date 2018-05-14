import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def draw():
	data = np.fromfile(sys.argv[1], dtype='float')
	N = int(math.sqrt(len(data)))
	data = [ data[x*N:(x+1)*N] for x in xrange(N) ]

	plt.imshow(data)
	plt.ylim(0,N)
	plt.colorbar(orientation='vertical')
	if(len(sys.argv) == 3 and sys.argv[2] == 'save'):
		plt.savefig(sys.argv[1][:-4]+'.png')
	else:
		plt.show()
if __name__ == '__main__':
	draw()
