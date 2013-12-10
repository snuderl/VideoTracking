import numpy as np
import numpy.random as random
from pygame import Rect
import time
from contextlib import contextmanager


@contextmanager
def measureTime(title):
    t1 = time.clock()
    yield
    t2 = time.clock()
    print '%s: %0.2f seconds elapsed' % (title, t2-t1)


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
    print "rectangleGenerator"
    print invalid
    l = []
    i = 0
    targetRect = Rect(invalid[0], invalid[1], invalid[2], invalid[3])
    while i < num:
        x1, y1 = random.uniform(0, height), random.uniform(0, width)
        h, w = random.uniform(5, height), random.uniform(5, width)
        #print h, w
        if h > 5 and w > 5:
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

