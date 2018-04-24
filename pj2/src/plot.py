import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

filename = sys.argv[1]

def get_data():
	data = np.fromfile(filename)
	N = int(math.sqrt(len(data)/20))
	data = [[data[t*N*N+x*N:t*N*N+(x+1)*N] for x in xrange(N) ] for t in xrange(20)]
	return data

def update(i):
	ax.cla()
	ax.imshow(data[i])
	ax.invert_yaxis()
	ax.set_title("frame {}".format(i))
	return ax

if __name__ == '__main__':
	data = get_data()

	fig, ax = plt.subplots()

	anim = FuncAnimation(fig, update, frames=np.arange(0,20), interval=100)

	if(len(sys.argv)==4 and sys.argv[3]=="save"):
		gif = filename.split('.')[0]+'.gif'
		anim.save(gif, writer='imagemagick')
	else:
		plt.show()

"""
python plot.py serial.out 200 [save]
"""