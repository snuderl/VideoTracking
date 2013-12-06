import particle
from matplotlib.pyplot import *
import numpy as np
import time
import matplotlib.collections as collections
import matplotlib.ticker as ticker



def animate(i):
	drawParticles(initial)
	particle.updateParticles(initial)

def drawParticles(particles):
	##Clear the screen
	cla()
	##Create collection of rectangles
	patches = []	
	for row in initial:
		patches.append(gca().add_patch(Rectangle((row[0], row[1]),row[2], row[3])))	
	p = collections.PatchCollection(patches)

	patches = ax.add_collection(p)
	draw()
	time.sleep(0.1)



initial = particle.createInitialParticles(np.array([70,50,20,20]), 10)
fig = figure()
ax=fig.add_subplot(111)
ax.set_xlim([0, 200]) #pylab.xlim([-400, 400])
ax.set_ylim([0, 200]) #pylab.ylim([-400, 400])
show(block=False)
for x in range(100):
	animate(x)