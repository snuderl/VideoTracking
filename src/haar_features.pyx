import numpy as np
from numpy import random
from matplotlib.pyplot import *
import matplotlib.collections as collections
from matplotlib.colors import ColorConverter
import cv2
import utils

from skimage.transform import integral_image

cimport cython
cimport numpy as np
# We represent a center surrond haar feature
# as an array of 4 points and a color selector
# [x1,y1,x2,y2,x3,y3,x4,y4]
# First 2 points represent the outer dimensions,
# while last 2 represent the inner dimensions
# of haar feature detecter
#
# Since all haar features will be translated to object space
# we define all ther coordinates in range [0-1] to faciliate this.


def generateHaarFeatures(number):
    prototype = np.array(
        [0, 0, 1, 1, 0.25, 0.25, 0.5, 0.5])

    scale = random.uniform(0.15, 0.6, (number, 1))

    protoypes = np.tile(prototype, (number, 1)) * scale

    translation = random.uniform(0, 0.8, (number, 2))
    protoypes[:, 0:2] += translation
    protoypes[:, 4:6] += translation

    # If generated features lies outside of range [0,1] translate it back
    # inside
    for row in protoypes:
        if row[0] + row[2] > 1:
            move = row[0] + row[2] - 1
            row[0] -= move
            row[4] -= move
        if row[1] + row[3] > 1:
            move = row[1] + row[3] - 1
            row[1] -= move
            row[5] -= move


    return protoypes

@cython.boundscheck(False)
def calculateFeatureVector(np.ndarray image,
                           np.ndarray particles,
                           np.ndarray haar_features,
                           np.ndarray target,
                           out=False,
                           indices=None):

    features = np.zeros((particles.shape[0],
                        haar_features.shape[0] * image.shape[2]))

    for i in range(particles.shape[0]):
        particle = particles[i,:]
        particle_image = cropImage(image, particle[0], particle[1], particle[2], particle[3])
        particle_image = integral_image(particle_image).astype(np.float32)
        calculated = calculateValues(
            particle_image, haar_features, indices).ravel()
        features[i, :] = calculated

    return features

cdef cropImage(np.ndarray image, float x,float y,float h,float w):
    return cv2.getRectSubPix(image,
                             (int(h), int(w)), (x + h / 2, y + w / 2))

# TODO: This function should be implemented using cv.remap,
# which would enable us to inteprolate without looping,
# or pershaps using numpys meshgrid
@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef np.ndarray[np.float32_t, ndim=1] calculateValues(np.ndarray image,np.ndarray haar_features, indices):
    '''This functions expects an integral image as input'''
    cdef int x1, y1, x2 ,y2
    cdef np.ndarray[np.float32_t, ndim=1] D,A,B,C
    cdef np.ndarray[np.float32_t, ndim=3] rectangle = image

    cdef int height = rectangle.shape[0]-1
    cdef int width = rectangle.shape[1]-1
    cdef int colors = rectangle.shape[2]
    cdef np.ndarray x = haar_features[:, ::2] * height
    cdef np.ndarray y = haar_features[:, 1::2] * width
    cdef np.ndarray[np.float32_t, ndim=1] areaOuter, areaInner, area
    cdef int c

    # rectangle = cv2.resize(rectangle, (50,50))
    # x = haar_features[:,::2]*25
    # y = haar_features[:,1::2]*25
    cdef np.ndarray coords1x = x[:, 0] + x[:, 1]
    cdef np.ndarray coords1y = y[:, 0] + y[:, 1]
    cdef np.ndarray coords2x = x[:, 2] + x[:, 3]
    cdef np.ndarray coords2y = y[:, 2] + y[:, 3]

    cdef np.ndarray[np.float32_t, ndim=1] values = np.zeros((haar_features.shape[0] * colors),
    dtype=np.float32).reshape(haar_features.shape[0] * colors)
    if not indices == None:
        for i in indices:
            x1, y1 = round(x[i / 3, 0]), round(y[i / 3, 0])
            x2, y2 = round(coords1x[i / 3]), round(coords1y[i / 3])

            D = rectangle[x2, y2]
            A = rectangle[x1, y1]
            B = rectangle[x2, y1]
            C = rectangle[x1, y2]
            areaOuter = D + A - B - C
            # print x1,y1,x2,y2
            # print D,A,B,C
            # print areaOuter

            x1, y1 = round(x[i / 3, 2]), round(y[i / 3, 2])
            x2, y2 = round(coords2x[i / 3]), round(coords2y[i / 3])
            D = rectangle[x2, y2]
            A = rectangle[x1, y1]
            B = rectangle[x2, y1]
            C = rectangle[x1, y2]
            areaInner = D + A - B - C

            c = i % 3
            area = areaOuter - 2 * areaInner
            # print c,i,values.shape, area
            values[i] = area[c]

    else:
        for i in range(haar_features.shape[0]):
            x1, y1 = round(x[i, 0]), round(y[i, 0])
            x2, y2 = round(coords1x[i]), round(coords1y[i])
            D = rectangle[x2, y2]
            A = rectangle[x1, y1]
            B = rectangle[x2, y1]
            C = rectangle[x1, y2]
            areaOuter = D + A - B - C
            # print x1,y1,x2,y2
            # print D,A,B,C
            # print areaOuter

            x1, y1 = round(x[i, 2]), round(y[i, 2])
            x2, y2 = round(coords2x[i]), round(coords2y[i])
            D = rectangle[x2, y2]
            A = rectangle[x1, y1]
            B = rectangle[x2, y1]
            C = rectangle[x1, y2]
            areaInner = D + A - B - C
            values[i*3:i*3 + 3] = (areaOuter - 2 * areaInner).ravel()
            # print x1,y1,x2,y2
            # print D,A,B,C
            # print areaInner

    return values


def visualizeHaarFeatures():
    fig = figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 1])  # pylab.xlim([-400, 400])
    ax.set_ylim([0, 1])  # pylab.ylim([-400, 400])
    patches = []

    cc = ColorConverter()
    outer = cc.to_rgba("#BFCBDE", alpha=0.5)
    inner = cc.to_rgba("RED", alpha=0.5)

    for row in generateHaarFeatures(100):
        patches.append(
            gca().add_patch(Rectangle((row[0], row[1]), row[2], row[3], color=outer)))
        patches.append(
            gca().add_patch(Rectangle((row[4], row[5]), row[6], row[7], color=inner)))
    p = collections.PatchCollection(patches)

    patches = ax.add_collection(p)

    show()


# print calculateValue(testImage, testHaar)


# print generateHaarFeatures(2)
if __name__ == "__main__":
    visualizeHaarFeatures()
    print testHaarFeatureCalculation()


def testHaarFeatureCalculation():
    test = np.array([[5, 2, 5, 2], [3, 6, 3, 6], [5, 2, 5, 2], [3, 6, 3, 6]])
    test2 = np.array([[5, 2], [3, 6]])
    integral = np.array([[0, 0, 0, 0, 0], [0, 5, 7, 12, 14], [0, 8, 16, 24, 32], [
                        0, 13, 23, 36, 46], [0, 16, 32, 48, 64]]).astype(np.float32).reshape((5, 5, 1))

    integral = cv2.integral(test.astype(np.float32)).astype(
        np.float32).reshape((5, 5, 1))
    feature = np.array([0, 0, 1, 1, 0.25, 0.25, 0.5, 0.5]).reshape(1, 8)
    #print calculateValues(integral, feature)
    print "--------------"
    integral = cv2.integral(test2.astype(np.float32)).astype(
        np.float32).reshape((3, 3, 1))

    feature = np.array([0, 0, 1, 1, 0.5, 0.5, 0.5, 0.5]).reshape(1, 8)
    #print calculateValues(integral, feature)