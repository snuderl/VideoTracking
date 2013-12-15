import cv2
import cv
import particle
import haar_features as haar
import numpy as np
import learner
import utils
import os
from utils import measureTime
from skimage.transform import integral_image
import pysomemodule
import signal

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

camera = False

directory = ""
ext = ".avi"
filename = "Vid_A_ball"
target = (200, 110, 50, 55)

#filename = "Vid_B_cup"
#target = (0.38960 * 320, 0.384615 * 240, 0.146011 * 320, 0.2440651 * 240)

#filename = "../data/Vid_D_person.avi"
#target = (0.431753*320, 0.240421*240, 0.126437 *320, 0.5431031*240)

if camera:
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(directory + filename + ext)

abc = pysomemodule.ABC("ab")

cv2.namedWindow("video")
cv2.namedWindow("particle")
cv2.namedWindow("test")

def onMouse(event, x, y, a, b):
    if event == 1:
        print event, x, y

cv2.setMouseCallback("video", onMouse, param=None)


def drawTarget(image, target):
    cv2.rectangle(image, (int(target[0]), int(target[1])), (
        int(target[0] + target[2]), int(target[1] + target[3])), 4)


def drawParticle(image, target):
    image = image.copy()
    image = utils.cropImage(image, target)
    cv2.imshow("particle", image)



from multiprocessing import Pool




def generateNewSamples(image, pf, features, pos, neg, newSamples=4):
    positive = pf.target.reshape((1, 4))
    generator = utils.rectangleGenerator(
        image.shape[1],
        image.shape[0],
        newSamples,
        invalid=pf.target)
    negative = np.array(list(generator))
    examples = np.vstack((negative, positive))

    image = image.astype(np.float32)
    features = features.astype(np.float32)
    examples = examples.astype(np.float32)

    feature_vector = np.nan_to_num(abc.doSomething(
        image.astype(np.float32), examples.astype(np.float32), features.astype(np.float32), allFeatures))
    #print feature_vector, feature_vector.dtype
    if not neg == None:
        if neg.shape[0] < 120:
            neg = np.vstack((neg, feature_vector[:-1, :]))
        else:
            neg[np.random.randint(0,neg.shape[0], (newSamples))] = feature_vector[:-1, :]
        positive = feature_vector[-1, :].reshape(1, feature_vector.shape[1])

        if pos.shape[0] >= 10:
            pos[np.random.randint(1,10), :] = positive
        else:
            pos = np.vstack((pos, positive))

    else:
        neg = feature_vector[:-1, :]
        positive = feature_vector[-1, :].reshape(1, feature_vector.shape[1])
        pos = positive

    return pos, neg

adaBoost = learner.Trainer(5)


def call(*args):
    return pysomemodule.ABC("ab").doSomething(*args)


pool = Pool(4, init_worker)
def iteration(image, pf, features, pos, neg, newSamples=5):
    image = image.astype(np.float32)

    with measureTime("Ada boost learning"):
        train = np.vstack((pos, neg))
        targets = np.zeros((pos.shape[0] + neg.shape[0]))
        targets[:pos.shape[0]] = 1  
        weights = np.ones(targets.shape)
        weights[0] = 4
        weights = weights / weights.sum()
        adaBoost.train(train, targets, weights)
        indices = adaBoost.features()
    print np.unique(indices % 3)

    with measureTime("Updating particles"):
        pf.updateParticles()
    with measureTime("Calculating particle features"):
        particles = pf.particles.astype(np.float32)
        size = particles.shape[0]/4
        r1 = pool.apply_async(call, [image, particles[:size,:], features.astype(np.float32), indices])
        r2 = pool.apply_async(call, [image, particles[size:2*size,:], features.astype(np.float32), indices])
        r3 = pool.apply_async(call, [image, particles[2*size:3*size,:], features.astype(np.float32), indices])
        r4 = pool.apply_async(call, [image, particles[3*size:,:], features.astype(np.float32), indices])
        particle_features = np.vstack((r1.get(), r2.get(), r3.get(), r4.get()))
        print particle_features.shape
    with measureTime("Scoring particles:"):
        scores = adaBoost.score(particle_features)[:, 1]
    with measureTime("Updatingh weights:"):
        pf.updateWeights(scores)
    with measureTime("Scoring target particle:"):
        targetVector = abc.doSomething(
        image, np.array([pf.target]).astype(np.float32), features.astype(np.float32), allFeatures)
        targetScore = adaBoost.score(targetVector)
        targetClass = adaBoost.predict(targetVector)
        print "Score {0}, ".format(targetScore)
    with measureTime("Generating new samples:"):
        pos, neg = generateNewSamples(
            image, pf, features, pos, neg, newSamples)

    return pos, neg


def start(image):
    directory = "out/{}".format(filename, iterationCount)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize particles
    pf = particle.ParticleFilter(target, 2000, image.shape[:2])
    # Generate haar features
    features = haar.generateHaarFeatures(featuresCount)
    #print features

    return target, pf, features

featuresCount = 120
allFeatures = np.array(list(range(0,featuresCount*3)))
iterationCount = 0
if __name__ == "__main__":
    try:
        pos, neg = None, None
        print pos
        if(capture.isOpened):
            #open("out/{}/output.txt".format(filename), "w")
            retval, image = capture.read()
            target, pf, features = start(image)
            pos, neg = generateNewSamples(image, pf, features, pos, neg, 100)
            pos, neg = iteration(image, pf, features, pos, neg)
            target = pf.target

        p = abc.test(image, target.astype(np.float32))
        cv2.imshow("test", p)
        while retval:


            cv2.imshow("video", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                pass
            retval, image = capture.read()
            iterationCount += 1
            with measureTime("Iteration {}".format(iterationCount)):
                pos, neg = iteration(image, pf, features, pos, neg)
            text = "Iteration {}".format(iterationCount)
            cv2.putText(image, text, (
                20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

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


            #directory =  "out/{}".format(filename)
            #if not os.path.exists(directory):
            #    os.makedirs(directory)
            cv2.imwrite(
                "out/{}/target{}.jpg".format(filename, iterationCount),
                utils.cropImage(image, target))

            drawTarget(image, target)
            drawParticle(image, target)

            cv2.imwrite(
                "out/{}/image{}.jpg".format(filename, iterationCount),
                image)







            # for x in pf.particles:
                #drawTarget(image, x)
            print "Iterations: {0}".format(pf.iterations)
    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        cv2.destroyWindow ("test")
        cv2.destroyWindow ("video")
        cv2.destroyWindow ("particle")
        pool.terminate()
        pool.join()
