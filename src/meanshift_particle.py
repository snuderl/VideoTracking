import config
import learner
import particle
import haar_features as haar
import numpy as np
import utils
from utils import measureTime
import cv2
import particle_utils

class MeanshiftParticleAlgorithm:
    def __init__(self):        
        self.kernelSize = 3
        self.kernelSigma = 2
        self.scale = 1
        self.kernelChanged = False
        pass

    @property
    def particles(self):
        return self.pf.particles

    @property
    def target(self):
        return self.pf.target

    def start(self, image, target):
        # Initialize particles
        self.iterations = 0
        self.roi = target
        self.initial = image
        self.image = image
        self.valid = True
        self.setup()
        self.pf = particle.ParticleFilter(
            target, 30, image.shape[:2], 0)

    def setup(self):
        image = self.initial
        roi = self.roi
        image = cv2.GaussianBlur(image, (self.kernelSize, self.kernelSize), self.kernelSigma)
        hsv =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        targetFrame = utils.cropImage(hsv, roi)
        mask = cv2.inRange(targetFrame, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([targetFrame], [0], mask, [180], [0,180])
        self.roi_hist = roi_hist

    def calculateHist(self, hsv, particle):     
        hist = cv2.calcBackProject(
            [hsv], [0], self.roi_hist, [0, 180], 1)
        return hist


    def next(self, image):
        self.image = image
        self.pf.updateParticles()

        if self.kernelChanged:
            self.setup()

        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 8, 1 )


        image = cv2.GaussianBlur(image, (self.kernelSize, self.kernelSize), self.kernelSigma)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject(
            [hsv], [0], self.roi_hist, [0, 180], 1)

        scores = []
        for particle in self.particles:
            x1, x2, x3, x4 = particle[:4].astype(np.int32)
            ret, track_window = cv2.meanShift(dst, (x1, x2, x3, x4), term_crit)
            particle[:2] = track_window[:2]
            track_window = particle[:4]

            f = utils.cropImage(hsv, track_window)
            mask = cv2.inRange(f, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            hist = cv2.calcHist([f], [0], mask, [180], [0,180])

            score = cv2.compareHist(hist.astype(np.float32), self.roi_hist.astype(np.float32), 3)
            scores.append(score)


        self.pf.updateWeights(1 - np.array(scores))
        self.best = self.target[:4].astype(np.int32)
        self.dst = dst
        print  max(scores), self.best[2:]

        self.iterations += 1


if __name__ == "__main__":
    algo = MeanshiftParticleAlgorithm()
    camera = False
    filename, target = utils.loadVideoFromFile("Vid_A_ball")
    if camera:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(filename)
    ret, frame = capture.read()
    algo.start(frame,  target)

    cv2.namedWindow("img2")
    cv2.namedWindow("options")
    particle_utils.initializeParticleSlider(algo.pf, "options", 10, 60, 0)
    particle_utils.initializeMeanshiftSlider(algo, "options")

    while(1):
        ret ,frame = capture.read()

        if ret == True:
            algo.next(frame)

            # Draw it on image
            x,y,w,h = algo.best
            cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
            particle_utils.drawParticles(frame, algo.pf)
            cv2.imshow('img2', frame)
            cv2.imshow('prob', algo.dst)

            k = cv2.waitKey(60) & 0xff
            if k == 27 or algo.iterations == 10000:
                img2.shape
                break

        else:
            break

    cv2.destroyAllWindows()
    cap.release()

