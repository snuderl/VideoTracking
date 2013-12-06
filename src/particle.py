import numpy as np
from numpy import random

SIGMA = 0.64

###State is represented as a numpy array 
### [x, y, w, h, vx, vy, weight]


def particleToString(particle):
	return "Weight: {6}, (x:{0}, y:{1}), (w:{2},h:{3}), (vx:{4},vy:{5})".format(*particle)

def createInitialParticles(target, count):
	'''Returns a matrix where each row represents the state of a particle'''
	velocities = random.normal(0, SIGMA, (count, 2))

	#Repeat target count times
	targets = np.tile(target, (count, 1))
	#Stack the rectangle, velocities and particle weight
	targets = np.hstack((targets,velocities, np.ones((count,1))/count))
	return targets

def updateParticles(particles):
	rows,cols = particles.shape
	#Update positions with velocity
	particles[:,0:2] += particles[:,4:6]
	#Add noise to w,h,vx,vy
	particles[:,2:6] += random.normal(0, SIGMA, (rows,4))

	###TODO update weight

def getTrackedObject(particles):
	rect = particles[:,0:4]
	weights = particles[:,6]
	weights = weights[None, :]
	##Calculates weight[i]*(x,y,w,h)[i] and sums it	
	return np.sum(rect * weights.T,axis=0)

initial = createInitialParticles(np.array([10,10,5,5]), 10)
print particleToString(initial[0,:])
print getTrackedObject(initial)



def particlefilter(initial, targer, stepsize, n):
  seq = iter(sequence)
  x = createInitialParticles(target, n)
  f0 = seq.next()[tuple(pos)] * ones(n)         # Target colour model
  yield pos, x, ones(n)/n                       # Return expected position, particles and weights
  for im in seq:
    x += uniform(-stepsize, stepsize, x.shape)  # Particle motion model: uniform step
    x  = x.clip(zeros(2), array(im.shape)-1).astype(int) # Clip out-of-bounds particles
    f  = im[tuple(x.T)]                         # Measure particle colours
    w  = 1./(1. + (f0-f)**2)                    # Weight~ inverse quadratic colour distance
    w /= sum(w)                                 # Normalize w
    yield sum(x.T*w, axis=1), x, w              # Return expected position, particles and weights
    if 1./sum(w**2) < n/2.:                     # If particle cloud degenerate:
      x  = x[resample(w),:]    

