import cv2
import cv
import os
from utils import measureTime
from algorithm import Algorithm
import utils


camera = False

directory = "../data/"
ext = ".avi"
filename = "Vid_A_ball"
target = (200, 110, 50, 55)

filename = "Vid_B_cup"
target = (0.38960 * 320, 0.384615 * 240, 0.146011 * 320, 0.2440651 * 240)

target = list(target)

#filename = "Vid_D_person"
#target = (0.431753*320, 0.240421*240, 0.126437 *320, 0.5431031*240)

#filename = "Vid_C_juice"
#target = (0.410029*320, 0.208388*240, 0.114061*320, 0.373526*240)


#filename = "Vid_E_person_partially_occluded"
#target = (0.434343*320, 0.18461*240, 0.167388*320, 0.675*240)

#filename = "Vid_C_juice"
#target = (0 * 320, 0.410029 * 240, 0.208388 * 320, 0.114061 * 240)




cv2.namedWindow("video")
cv2.namedWindow("particle")
#cv2.namedWindow("test")

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1


# mouse callback function
def onMouse(event,x,y,flags,param):
    global target, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        target[0] = x
        target[1] = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            
            target[2] = x-target[0]
            target[3] = y-target[1]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
        target[2] = x-target[0]
        target[3] = y-target[1]

def onStart(x):
    global mode
    if x == 0:
        mode = setupMode
    elif x==1:
        algo.start(image, target)
        mode = algoMode


drawParticles = False
def drawParticlesState(x):
    global drawParticles    
    if x == 0:
        drawParticles = False
    elif x==1:
        drawParticles = True

cv2.setMouseCallback("video", onMouse, param=None)
cv2.createTrackbar("Setup - Run", "video", 0, 1, onStart)
cv2.createTrackbar("Draw particles", "video", 0, 1, drawParticlesState)

def algoMode():
    global target
    global image

    with measureTime("Iteration {}".format(iterationCount)):
        algo.next(image)
        text = "Iteration {}".format(iterationCount)
        cv2.putText(image, text, (
            20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        target = algo.target

        utils.drawTarget(image, target)
        targetImage = utils.drawParticle(image, target)
        if drawParticles:
            algo.drawFeatures(targetImage)
        targetImage = cv2.resize(targetImage, (targetImage.shape[1]*2, targetImage.shape[0]*2))
        cv2.imshow("particle", targetImage)        
        print "Frame: {0}".format(algo.iterations)

def setupMode():
    global target
    global image

    utils.drawTarget(image, target)


iterationCount = 0

if camera:
    capture = cv2.VideoCapture(0)
    started = False
    mode = setupMode
else:
    capture = cv2.VideoCapture(directory + filename + ext)
    started = True
    mode = algoMode

def write(name, locations):
    with open(name, "w") as f:
        f.write("./" + filename + ".avi\n")
        for i, target in enumerate(locations):
            f.write("{} {} {} {} {}\n".format(i, *target))


locations = []
if __name__ == "__main__":
    try:
        algo = Algorithm(1)
        if(capture.isOpened):
            retval, image = capture.read()
            w,h = image.shape[0], image.shape[1]
            if camera:
                image = cv2.flip(image, 1)
            else:
                algo.start(image, target)
        while retval:
            with measureTime("Frame processed in"):
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c'):
                    pass
                retval, image = capture.read()
                if not retval:
                    break
                if camera:
                    image = cv2.flip(image, 1)
                else:                    
                    locations.append(algo.target / [h, w, h, w])
                iterationCount += 1
                mode()
                    
                if drawParticles:
                    algo.drawParticles(image)
                cv2.imshow("video", image)

        write("../data/Tracked/" + filename + ".txt", locations)
    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        cv2.destroyWindow ("test")
        cv2.destroyWindow ("video")
        cv2.destroyWindow ("particle")
        if algo.pool:
            self.pool.terminate()
            self.pool.join()
