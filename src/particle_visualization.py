import particle
from matplotlib.pyplot import *
import numpy as np
import time
import matplotlib.collections as collections
import matplotlib.ticker as ticker

def score(center, particles):
	dx = (center[0]-particles[:,0])**2 
	dy = (center[1]-particles[:,1])**2
	return np.sqrt(1/(dx+dy))

def animate(i):
	initial.updateParticles()
	center = (70,50)
	r = 50
	t = i/20.
	centerx = r*np.sin(t)+center[0]
	centery = r*np.cos(t)+center[1]
	drawParticles(initial, (centerx,centery))


	scores = score((centerx,centery), initial.particles)
	initial.updateWeights(scores)
	print "Resampled"
	#drawParticles(initial)
	#time.sleep(0.5)

def drawParticles(pf, center=None):
	##Clear the screen
	cla()
	##Create collection of rectangles
	patches = []	
	for row in pf.particles:
		patches.append(Rectangle((row[0], row[1]),row[2], row[3]))	
	if center:
		patches.append(Rectangle(center,2, 2, color="red"))
	p = collections.PatchCollection(patches)

	patches = ax.add_collection(p)
	draw()
	time.sleep(0.05)



initial = particle.ParticleFilter(np.array([70,100,20,20]), 600, (200,200))
fig = figure()
ax=fig.add_subplot(111)
ax.set_xlim([0, 200]) #pylab.xlim([-400, 400])
ax.set_ylim([0, 200]) #pylab.ylim([-400, 400])
show(block=False)
for x in range(100):
	animate(x)