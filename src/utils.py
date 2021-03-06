import numpy as np
import numpy.random as random
from pygame import Rect
import time
import cv2
from contextlib import contextmanager
from os import listdir
from os.path import isfile, join

def loadVideoFromFile(name, directory = "../data/", ext=".avi"):

    files = [f for f in listdir(directory) if isfile(join(directory,f))]
    txtFile = [f for f in files if f.startswith(name) and f.endswith(".txt")][0]
    with open(join(directory, txtFile)) as f:
        line = f.readlines()[1]
        coordinates = map(float, line.split(" ")[1:])
    x,y,w,h = coordinates
    return join(directory, [f for f in files if f.startswith(name) and f.endswith(".avi")][0]), (x * 320, y * 240, w * 320, h * 240)

def drawParticle(image, target):
    image = image.copy()
    image = cropImage(image, target)
    return image

def write(name, filename, locations):
    with open(name, "w") as f:
        f.write("./" + filename + ".avi\n")
        for i, target in enumerate(locations[:-1]):
            f.write("{} {} {} {} {}\n".format(i, *target))

def drawTarget(image, target, color=(0,0,0)):
    cv2.rectangle(image, (int(target[0]), int(target[1])), (
        int(target[0] + target[2]), int(target[1] + target[3])), color)

@contextmanager
def measureTime(title, switch=True):
    t1 = time.time()
    yield
    t2 = time.time()
    if switch:
        print '%s: %0.2f seconds elapsed' % (title, t2-t1)

def cropImage(image, rectangle):
    x, y = rectangle[0], rectangle[1]
    h, w = rectangle[2], rectangle[3]
    return cv2.getRectSubPix(image,
                             (int(h), int(w)), (x + h / 2, y + w / 2))

def timeit(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        value = fn(*args, **kwargs)
        print "Elapsed {0} in function {1}".\
            format(time.time() - start,  fn.__name__)
        return value
    return wrapper


def rectangleGenerator(height,
                       width,
                       num,
                       invalid=[-100, -100, 1, 1]):
    #print "rectangleGenerator"
    #qprint invalid
    l = []
    i = 0
    targetRect = Rect(invalid[0], invalid[1], invalid[2], invalid[3])
    while i < num:
        x1, y1 = random.uniform(0, height), random.uniform(0, width)
        h, w = invalid[2] + random.uniform(-30, 30), invalid[3] + random.uniform(-30, 30)
        #print h, w
        if h > 5 and w > 5 and h < height and w < width:    
            rectangle = np.array([x1, y1, h, w])
            test = Rect(x1, y1, h, w)
            if not targetRect.colliderect(test):
                l.append(rectangle)
                i += 1
    return l


def isRectangleOverlaping(target, rectangle):
    if abs(target[0] - rectangle[0]) < abs(target[2] - rectangle[2]):
        return False
    if abs(target[1] - rectangle[1]) < abs(target[3] - rectangle[3]):
        return False
    return True


if __name__ == "__main__":
    import cv2
    filename = "../data/Vid_A_ball.avi"

    capture = cv2.VideoCapture(filename)
    r, img = capture.read()
    cv2.rectangle(img, (200,110), (200+50,110+55), 200)
    rectangles = rectangleGenerator(img.shape[1],img.shape[0], 50, invalid=[200,110,50.,55.])
    for x in rectangles:
        cv2.rectangle(img, (int(x[0]),int(x[1])), (int(x[0]+x[2]),int(x[1]+x[3])),5)
    while True:

        cv2.imshow("video", img)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

