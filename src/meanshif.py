import config
import learner
import particle
import haar_features as haar
import numpy as np
import utils
from utils import measureTime
import cv2
import particle_utils


class Meanshift(object):
    def __init__(self):
        self.kernelSize = 3
        self.kernelSigma = 2
        pass

    def start(self, image, target):
        # Initialize particles
        self.iterations = 0
        self.initial = image
        self.roi = target
        self.target = target
        self.setup()
        self.scale = 1
        self.valid = True
        self.kernelChanged = False
        self.pf = particle.ParticleFilter(
            target, 30, image.shape[:2])

    def setup(self):
        roi = self.roi
        image = self.initial
        image = cv2.GaussianBlur(image, (self.kernelSize, self.kernelSize), self.kernelSigma)
        hsv =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        targetFrame = utils.cropImage(hsv, roi)
        mask = self.getMask(targetFrame)
        roi_hist = cv2.calcHist([targetFrame], [0], mask, [180], [0,180])
        self.initial_hist = roi_hist
        self.roi_hist = roi_hist

    def getMask(self, image):
        return cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

    def hist(self, hsv_image):
        mask = self.getMask(hsv_image)
        return cv2.calcHist([hsv_image], [0], mask, [180], [0,180])



    def next(self, image):
        self.image = image

        if self.kernelChanged:
            self.setup()


        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1 )


        image = cv2.GaussianBlur(image, (self.kernelSize, self.kernelSize), self.kernelSigma)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject(
            [hsv], [0], self.roi_hist, [0, 180], self.scale)

        ret, track_window = cv2.meanShift(dst, tuple(map(int, self.target)), term_crit)

        self.target = track_window

        window = utils.cropImage(hsv, track_window)
        targetFrame = utils.cropImage(hsv, track_window)
        window = targetFrame
        hist = self.hist(window)
        #self.roi_hist = (self.initial_hist + hist) / 2
        self.dst = dst
        print self.roi_hist.shape, dst.shape, dst.dtype

        self.iterations += 1


if __name__ == "__main__":
    camera = False

    algo = Meanshift()


    filename, target = utils.loadVideoFromFile("Vid_A_ball")
    if camera:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(filename)
    ret ,frame = capture.read()

    cv2.namedWindow("img2")
    particle_utils.initializeMeanshiftSlider(algo, "img2")

    algo.start(frame,  target)
    while(1):
        ret ,frame = capture.read()
        if ret == True:
            algo.next(frame)

            # Draw it on image
            x,y,w,h = algo.target
            cv2.rectangle(frame, (int(x),int(y)), (int(x+w),int(y+h)), 255,2)
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