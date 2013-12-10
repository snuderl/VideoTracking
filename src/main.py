import cv2
import particle
import haar_features as haar
import numpy as np
import learner
import utils
import os
from utils import measureTime

filename = "../data/Vid_A_ball.avi"
target = (200, 110, 50, 55)

# filename = "../data/Vid_B_cup.avi"
# target = (0.38960*320, 0.384615*240, 0.146011*320, 0.2440651*240)

# filename = "../data/Vid_D_person.avi"
# target = (0.431753*320, 0.240421*240, 0.126437 *320, 0.5431031*240)

capture = cv2.VideoCapture(filename)

cv2.namedWindow("video")
cv2.namedWindow("particle")



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
    x, y = rectangle[0], rectangle[1]
    h, w = rectangle[2], rectangle[3]
    return cv2.getRectSubPix(image.astype(np.uint8),
                             (int(h), int(w)), (x+h/2, y+h/2))




def calculateFeatureVector(image,
                           particles,
                           haar_features,
                           target,
                           out=False,
                           indices=None):

    integral = cv2.integral(image)
    #print integral
    features = np.zeros((particles.shape[0],
                        haar_features.shape[0]*image.shape[2]))
    cv2.imwrite("out/target{}.jpg".format(iterationCount),
                      cropImage(image, target))
    for i, particle in enumerate(particles):
        particle_image = cropImage(integral, particle)
        calculated = haar.calculateValues(particle_image, haar_features, indices=indices).ravel()
        #print calculated.shape
        #print calculated.shape
        features[i, :] = calculated
        if out:
            directory = "out/iteration{}".format(iterationCount)
            if not os.path.exists(directory):
                os.makedirs(directory)
            particle_image_orig = cropImage(image, particle)
            name = "/rect{}.jpg".format(i)
            cv2.imwrite(directory+name, particle_image_orig)

    return features

def generateNewSamples(image, pf, features, pos, neg, newSamples=5):
    positive = pf.target.reshape((1, 4))
    generator = utils.rectangleGenerator(
        image.shape[1],
        image.shape[0],
        newSamples,
        invalid=pf.target)
    negative = np.array(list(generator))
    examples = np.vstack((negative, positive))

    feature_vector = calculateFeatureVector(image, examples, features, pf.target, out=False)
    if not neg == None:
        neg = np.vstack((neg, feature_vector[:-1,:]))
        pos = np.vstack((pos, feature_vector[-1,:].reshape(1,450)))

    else:
        neg = feature_vector[:-1,:]
        pos = feature_vector[-1,:].reshape(1,450)

    return pos, neg


def iteration(image, pf, features, pos, neg, newSamples=5):
    train = np.vstack((pos, neg))
    targets = np.zeros((pos.shape[0]+neg.shape[0]))
    targets[:pos.shape[0]]=1

    adaBoost = learner.initialize(32)
    with measureTime("Ada boost learning"):
        adaBoost.fit(train, targets)
        indices = adaBoost.feature_importances_.argsort()[-32:][::-1]
        #print targets

    #print feature_vector
    #probabilities = adaBoost.predict_proba(feature_vector)[:,1]

    with measureTime("Updating particles"):
        pf.updateParticles()
    with measureTime("Calculating particle features"):
        particle_features = calculateFeatureVector(image, pf.particles, features, pf.target, indices=indices)
    scores = adaBoost.predict_proba(particle_features)[:,1]
    print adaBoost.predict_proba(particle_features), adaBoost.predict(particle_features).sum()
    pf.updateWeights(scores)

    #drawParticle(image, pf.particles[scores.argmax()])

    #print np.max(scores)
    #print particles[:,6]
    with measureTime("Generating new samples:"):
        pos, neg = generateNewSamples(image, pf, features, pos, neg, newSamples)

    return pos, neg


def start(image):
    ###Initialize particles
    pf = particle.ParticleFilter(target, 2000, image.shape[:2])
    ###Generate haar features
    features = haar.generateHaarFeatures(150)

    return target, pf, features


iterationCount = 0
if __name__ == "__main__":
    pos, neg = None, None
    if(capture.isOpened):
        retval, image = capture.read()
        target, pf, features = start(image)
        pos, neg = generateNewSamples(image, pf, features, pos, neg, 100)
        target = pf.target
    while retval:

        cv2.imshow("video", image)
        key = cv2.waitKey(500) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            pass
        retval, image = capture.read()
        iterationCount += 1
        with measureTime("Iteration {}".format(iterationCount)):
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

        #for x in pf.particles:
            #drawTarget(image, x)
        print "Iterations: {0}".format(pf.iterations)
