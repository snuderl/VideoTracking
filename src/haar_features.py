import numpy as np
from numpy import random
from matplotlib.pyplot import *
import matplotlib.collections as collections
from matplotlib.colors import ColorConverter
import cv2

from skimage.transform import integral_image

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

# TODO: This function should be implemented using cv.remap,
# which would enable us to inteprolate without looping,
# or pershaps using numpys meshgrid
#from numba import autojit

#@autojit
def calculateValues(rectangle, haar_features, indices):
    '''This functions expects an integral image as input'''

    rectangle = integral_image(rectangle).astype(np.float32)
    #print rectangle.dtype
    #print rectangle.dtype
    #print rectangle.shape
    height, width, colors = rectangle.shape
    x = haar_features[:, ::2] * rectangle.shape[0]
    y = haar_features[:, 1::2] * rectangle.shape[1]

    # rectangle = cv2.resize(rectangle, (50,50))
    # x = haar_features[:,::2]*25
    # y = haar_features[:,1::2]*25
    coords1x = x[:, 0] + x[:, 1]/2
    coords1y = y[:, 0] + y[:, 1]/2
    coords2x = x[:, 2] + x[:, 3]/2
    coords2y = y[:, 2] + y[:, 3]/2


    values = np.zeros((haar_features.shape[0] * colors))
    if not indices == None:
        for i in indices:
            x1, y1 = x[i / 3, 0], y[i / 3, 0]
            x2, y2 = coords1x[i / 3], coords1y[i / 3]

            D = cv2.getRectSubPix(rectangle, (1, 1), (x2, y2))
            A = cv2.getRectSubPix(rectangle, (1, 1), (x1, y1))
            B = cv2.getRectSubPix(rectangle, (1, 1), (x1, y2))
            C = cv2.getRectSubPix(rectangle, (1, 1), (x2, y1))
            areaOuter = D + A - B - C
            # print x1,y1,x2,y2
            # print D,A,B,C
            # print areaOuter

            x1, y1 = x[i / 3, 2], y[i / 3, 2]
            x2, y2 = coords2x[i / 3], coords2y[i / 3]
            D = cv2.getRectSubPix(rectangle, (1, 1), (x2, y2))
            A = cv2.getRectSubPix(rectangle, (1, 1), (x1, y1))
            B = cv2.getRectSubPix(rectangle, (1, 1), (x1, y2))
            C = cv2.getRectSubPix(rectangle, (1, 1), (x2, y1))
            areaInner = D + A - B - C

            c = i % 3
            area = areaOuter - 2 * areaInner
            # print c,i,values.shape, area
            values[i] = area[0][0][c]

    else:
        for i in range(haar_features.shape[0]):
            x1, y1 = x[i, 0], y[i, 0]
            x2, y2 = coords1x[i], coords1y[i]
            D = cv2.getRectSubPix(rectangle, (1, 1), (x2, y2))
            A = cv2.getRectSubPix(rectangle, (1, 1), (x1, y1))
            B = cv2.getRectSubPix(rectangle, (1, 1), (x1, y2))
            C = cv2.getRectSubPix(rectangle, (1, 1), (x2, y1))
            areaOuter = D + A - B - C
            # print x1,y1,x2,y2
            # print D,A,B,C
            # print areaOuter

            x1, y1 = x[i, 2], y[i, 2]
            x2, y2 = coords2x[i], coords2y[i]
            D = cv2.getRectSubPix(rectangle, (1, 1), (x2, y2))
            A = cv2.getRectSubPix(rectangle, (1, 1), (x1, y1))
            B = cv2.getRectSubPix(rectangle, (1, 1), (x1, y2))
            C = cv2.getRectSubPix(rectangle, (1, 1), (x2, y1))
            areaInner = D + A - B - C
            values[i:i+3] = (areaOuter - 2 * areaInner).ravel()
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
    print calculateValues(integral, feature)
    print "--------------"
    integral = cv2.integral(test2.astype(np.float32)).astype(
        np.float32).reshape((3, 3, 1))

    feature = np.array([0, 0, 1, 1, 0.5, 0.5, 0.5, 0.5]).reshape(1, 8)
    print calculateValues(integral, feature)
