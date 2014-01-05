import cv2
import cv
import os
from utils import measureTime
from algorithm import Algorithm
import utils
import particle_utils



camera = False



filename = "Vid_B_cup"
filename = "Vid_I_person_crossing"
filename = "Vid_D_person"
filename = "Vid_C_juice"
filename = "Vid_C_juice"
filename = "Vid_B_cup"


filename, target = utils.loadVideoFromFile("Vid_A")


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




cv2.setMouseCallback("video", onMouse, param=None)
cv2.createTrackbar("Setup - Run", "video", 0, 1, onStart)

def algoMode():
    global target
    global image
    algo.next(image)
    text = "Iteration {}".format(iterationCount)
    cv2.putText(image, text, (
        20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    target = algo.target

    color = (255, 0, 0)
    if algo.valid:
        color = (0, 255, 0)

    utils.drawTarget(image, target, color)

    targetImage = utils.drawParticle(image, target)

    targetImage = cv2.resize(targetImage, (targetImage.shape[1]*2, targetImage.shape[0]*2))
    cv2.imshow("particle", targetImage)        

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
    capture = cv2.VideoCapture(filename)
    started = True
    mode = algoMode



locations = []
if __name__ == "__main__":
    try:
        algo = Algorithm(1)

        cv2.namedWindow("options")
        if(capture.isOpened):
            retval, image = capture.read()
            w,h = image.shape[0], image.shape[1]
            if camera:
                image = cv2.flip(image, 1)
            else:
                algo.start(image, target)
                particle_utils.initializeParticleSlider(algo.pf, "options")
                particle_utils.drawParticles(image, algo.pf)
                cv2.imshow("video", image)

        while True:

            key = cv2.waitKey(10) & 0xFF 
            if key == ord("c"):
                break

        while retval:
            with measureTime("Frame processed in", True):
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
                    
                particle_utils.drawParticles(image, algo.pf)
                cv2.imshow("video", image)

        #utils.write("../data/Tracked/" + filename + ".txt", filename, locations)
    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        cv2.destroyWindow ("test")
        cv2.destroyWindow ("video")
        cv2.destroyWindow ("particle")
        if algo.pool:
            self.pool.terminate()
            self.pool.join()
