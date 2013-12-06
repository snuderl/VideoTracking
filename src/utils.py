import numpy as np
import numpy.random as random


def rectangleGenerator(height,width):
	print "rectangleGenerator"
	while True:
		x1,x2 = random.uniform(0,height,(2,1))
		y1,y2 = random.uniform(0,width,(2,1))
		x1,x2 = min(x1[0],x2[0]), max(x1[0],x2[0])
		y1,y2 = min(y1[0],y2[0]), max(y1[0],y2[0])
		h,w = x2-x1,y2-y1
		#print h, w
		if h > 5 and w > 5:
			yield np.array([x1,y1,x2-x1,y2-y1])


if __name__ == "__main__":
	i = 0
	for x in rectangleGenerator(100,100):
		if i == 50:
			break
		print x
		i+=1