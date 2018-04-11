import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


data = []
filename = sys.argv[1]

def get_data():
	with open(filename, "r") as file:
		lines = file.readlines()
		for line in lines:
			array = []
			for str in line.split():
				array.append(float(str))
			data.append(array)
		file.close()
		return len(data[0])

def update(i):
	ax.cla()
	ax.imshow(data[i*N:(i+1)*N])
	ax.invert_yaxis()
	ax.set_title("frame {}".format(i))
	return ax

if __name__ == '__main__':
	N = get_data()

	fig, ax = plt.subplots()

	anim = FuncAnimation(fig, update, frames=np.arange(0,len(data)/N), interval=100)

	if(len(sys.argv)==3 and sys.argv[2]=="save"):
		gif = filename.split('.')[0]+'.gif'
		anim.save(gif, writer='imagemagick')
	else:
		plt.show()