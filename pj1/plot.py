import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

filename = sys.argv[1]
N = int(sys.argv[2])
steps = int(sys.argv[3])

def read_frame():
	matrix = []
	for i in xrange(N):
		array = []
		for str in file.readline().split():
			array.append(float(str))
		matrix.append(array)
	return matrix

def update(i):
	ax.cla()
	ax.imshow(data[i])
	ax.invert_yaxis()
	ax.set_title("frame {}".format(i))
	return ax

data = []
with open(filename, "r") as file:
	for i in xrange(steps):
		data.append(read_frame())
	file.close()

if __name__ == '__main__':
	fig, ax = plt.subplots()

	anim = FuncAnimation(fig, update, frames=np.arange(0,steps), interval=100)

	gif = filename.split('.')[0]+'.gif'
	anim.save(gif, writer='imagemagick')