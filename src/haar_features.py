import numpy as np
from numpy import random
from matplotlib.pyplot import *
import matplotlib.collections as collections
from matplotlib.colors import ColorConverter
import cv2


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

    protoypes[:,2:4] = protoypes[:,2:4] + protoypes[:,0:2]
    protoypes[:,6:8] = protoypes[:,6:8] + protoypes[:,4:6]

    for row in protoypes:
        for i in range(row.shape[0]):
            if(row[i] > 1):
                row[i] = 1

    
    return protoypes.astype(np.float32)

# TODO: This function should be implemented using cv.remap,
# which would enable us to inteprolate without looping,
# or pershaps using numpys meshgrid
#from numba import autojit

#@autojit


def calculateValues(rectangle, haar_features, indices):
    '''This functions expects an integral image as input'''

    height, width, colors = rectangle.shape
    height -= 1
    width -= 1
    x = np.round(haar_features[:,0] * height).astype(np.int32)
    y = np.round(haar_features[:,1] * width).astype(np.int32)
    values = rectangle[x,y]

    test = values.reshape((values.shape[0]/8, 8, 3))
    values = test[:,0,:]+test[:,1,:]-test[:,2,:]-test[:,3,:]-2*\
    (test[:,4,:]+test[:,5,:]-test[:,6,:]-test[:,7,:])
    return values.ravel()


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
        #print row
        patches.append(
            gca().add_patch(Rectangle((row[0], row[1]), row[2]-row[0], row[3]-row[1], color=outer)))
        patches.append(
            gca().add_patch(Rectangle((row[4], row[5]), row[6]-row[4], row[7]-row[5], color=inner)))
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
