import cv2
import particle
import haar_features as haar
import numpy as np
import time
from itertools import islice
import learner
import utils

filename = "../data/Vid_A_ball.avi"

capture = cv2.VideoCapture(filename)

cv2.namedWindow("video")
cv2.namedWindow("particle")

def timeit(fn):
    def wrapper(*args,**kwargs):
        start = time.time()
        value = fn(*args,**kwargs)
        print "Elapsed {0}".format(time.time()-start)
        return value
    return wrapper

def onMouse(event, x,y, a, b):
    if event==1:
        print event,x,y

cv2.setMouseCallback("video", onMouse, param=None) 


def drawTarget(image, target):
    cv2.rectangle(image, (int(target[0]), int(target[1])), (int(target[0]+target[2]), int(target[1]+target[3])), 4)

def drawParticle(image, target):
    image = image.copy()
    image = cropImage(image, target)
    cv2.imshow("particle", image)

def cropImage(image, rectangle):
    x,y = rectangle[0], rectangle[1]
    h,w = rectangle[2], rectangle[3]
    return cv2.getRectSubPix(image.astype(np.uint8), (int(h),int(w)), (x+h/2, y+h/2))

def calculateFeatureVector(image, particles, haar_features):
    integral = cv2.integral(image)
    #print integral
    features = np.zeros((particles.shape[0], haar_features.shape[0]*image.shape[2]))
    for i, particle in enumerate(particles):

        #print x,y,h,w

        particle_image = cropImage(integral, particle)
        calculated = haar.calculateValues(particle_image, haar_features).ravel()
        #print calculated.shape
        #print calculated.shape
        features[i, :] = calculated

        #print particle_image.shape


    # ###Test
    #
    #print particle_image
    #print particle_image.shape
    #print "particle image"
    #img = cv2.getRectSubPix(image, (int(h),int(w)), (x+h/2, y+h/2))
    #cv2.imshow("particle", particle_image)
    # print h,w
    # print particle_image.shape
    #print features.shape
    return features

@timeit
def iteration(image, pf, features, pos, neg, newSamples=5):
    positive = pf.getTrackedObject().reshape((1,4))
    negative = np.array([rect for rect in islice(
        utils.rectangleGenerator(image.shape[0], image.shape[1]),
        0,newSamples)])


    examples = np.vstack((negative,positive))
    #print "examples"
    #print examples.shape

    feature_vector = calculateFeatureVector(image, examples, features)
    if not neg == None:
        #print neg.shape, feature_vector.shape
        #print neg.shape, pos.shape
        neg = np.vstack((neg, feature_vector[:-1,:]))
        pos = np.vstack((pos, feature_vector[-1,:].reshape(1,300)))
        #print neg.shape, pos.shape

    else:
        neg = feature_vector[:-1,:]
        pos = feature_vector[-1,:].reshape(1,300)



    train = np.vstack((pos, neg))
    targets = np.zeros((pos.shape[0]+neg.shape[0]))
    #print pos.shape[0]
    targets[:pos.shape[0]]=1


    #print targets.shape, train.shape
    #print pos.shape, neg.shape
    #print positives

    adaBoost = learner.initialize(32)
    adaBoost.fit(train, targets)
    #print feature_vector
    #probabilities = adaBoost.predict_proba(feature_vector)[:,1]


    pf.updateParticles()
    particle_features = calculateFeatureVector(image, pf.particles, features)
    scores = adaBoost.predict_proba(particle_features)[:,1]
    pf.updateWeights(scores)
    #print np.max(scores)
    #print particles[:,6]
    return pos, neg










def start(image):
    target = (200,110,50,55)
    ###Initialize particles
    pf = particle.ParticleFilter(target, 200, image.shape[:2])
    ###Generate haar features
    features = haar.generateHaarFeatures(100)

    return target, pf, features



if __name__ == "__main__":
    pos, neg = None, None
    if(capture.isOpened):
        retval, image = capture.read()
        target, pf, features = start(image)
        pos, neg = iteration(image, pf, features, pos, neg, newSamples=100)
        target = pf.target
    while retval:
        
        cv2.imshow("video", image)
        key = cv2.waitKey(500) & 0xFF 
        if key == ord('q'):
            break
        if key == ord('c'):
            pass
        retval, image = capture.read()
        pos, neg = iteration(image, pf, features, pos, neg)
        # for p in pf.particles:
        #     drawParticle(image, p)   
        #     displayParticles = True     
        #     while displayParticles:        
        #         key = cv2.waitKey(500) & 0xFF 
        #         if key == ord('c'):
        #             break
        #         if key == ord('q'):
        #             displayParticles = False
        #     if not displayParticles:
        #         break
        target = pf.target
        drawTarget(image, target)
        drawParticle(image, target)
        for x in pf.particles:
            drawTarget(image, x)
        print "Iterations: {0}".format(pf.iterations)