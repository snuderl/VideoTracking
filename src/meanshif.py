import config
import learner
import particle
import haar_features as haar
import numpy as np
import utils
from utils import measureTime
import cv2


class Meanshift(object):
    def __init__(self):
        self.kernelSize = 3
        self.sigma = 2
        pass

    def start(self, image, target):
        # Initialize particles
        self.iterations = 0
        self.image = image
        self.target = target
        self.setup(image, target)
        self.pf = particle.ParticleFilter(
            target, 30, image.shape[:2])

    def setup(self, image, roi):
        image = cv2.GaussianBlur(image, (self.kernelSize, self.kernelSize), self.sigma)
        hsv =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        targetFrameHsv = utils.cropImage(hsv, roi)
        targetFrameRGB = utils.cropImage(image, roi)
        targetFrame = np.zeros((targetFrameHsv.shape[0], targetFrameHsv.shape[1], 6), image.dtype)
        targetFrame[:,:,:3] = targetFrameHsv
        targetFrame[:,:,3:] = targetFrameRGB
        mask = self.getMask(targetFrame)
        roi_hist = cv2.calcHist([targetFrame], [0], mask, [180], [0,180])
        self.initial_hist = roi_hist
        self.roi_hist = roi_hist

    def getMask(self, image):
        return cv2.inRange(image, np.array((0., 60.,32.,10.,10.,10.)), np.array((180.,255.,255.,255.,255.,255.)))

    def hist(self, hsv_image):
        mask = self.getMask(hsv_image)
        return cv2.calcHist([hsv_image], [0], mask, [180], [0,180])



    def next(self, image):
        self.image = image


        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1 )


        image = cv2.GaussianBlur(image, (self.kernelSize, self.kernelSize), self.sigma)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_hsv = np.zeros((hsv.shape[0], hsv.shape[1], 6), image.dtype)
        color_hsv[:,:,:3] = hsv
        color_hsv[:,:,3:] = image
        dst = cv2.calcBackProject(
            [color_hsv], [0], self.roi_hist, [0, 180], 1)

        ret, track_window = cv2.meanShift(dst, tuple(map(int, self.target)), term_crit)

        self.target = track_window

        window = utils.cropImage(hsv, track_window)
        targetFrameHsv = utils.cropImage(hsv, track_window)
        targetFrameRGB = utils.cropImage(image, track_window)
        targetFrame = np.zeros((targetFrameHsv.shape[0], targetFrameHsv.shape[1], 6), image.dtype)
        targetFrame[:,:,:3] = targetFrameHsv
        targetFrame[:,:,3:] = targetFrameRGB
        window = targetFrame
        hist = self.hist(window)
        #self.roi_hist = (self.initial_hist + hist) / 2
        self.dst = dst
        print self.roi_hist.shape, dst.shape, dst.dtype

        self.iterations += 1


camera = False

algo = Meanshift()


filename, target = utils.loadVideoFromFile("Vid_C_juice")
if camera:
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(filename)
ret ,frame = capture.read()


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