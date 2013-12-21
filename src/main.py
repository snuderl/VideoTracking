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



#filename = "Vid_D_person"
#target = (0.431753*320, 0.240421*240, 0.126437 *320, 0.5431031*240)

#filename = "Vid_C_juice"
#target = (0.410029*320, 0.208388*240, 0.114061*320, 0.373526*240)


#filename = "Vid_E_person_partially_occluded"
#target = (0.434343*320, 0.18461*240, 0.167388*320, 0.675*240)

#filename = "Vid_C_juice"
#target = (0 * 320, 0.410029 * 240, 0.208388 * 320, 0.114061 * 240)

if camera:
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture(directory + filename + ext)



cv2.namedWindow("video")
cv2.namedWindow("particle")
#cv2.namedWindow("test")

def onMouse(event, x, y, a, b):
    if event == 1:
        print event, x, y

cv2.setMouseCallback("video", onMouse, param=None)


iterationCount = 0
if __name__ == "__main__":
    try:
        algo = Algorithm(1)
        if(capture.isOpened):
            retval, image = capture.read()
            algo.start(image, target)
        while retval:
            with measureTime("Frame processed in"):
                cv2.imshow("video", image)
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c'):
                    pass
                retval, image = capture.read()
                iterationCount += 1
                with measureTime("Iteration {}".format(iterationCount)):
                    algo.next(image)
                text = "Iteration {}".format(iterationCount)
                cv2.putText(image, text, (
                    20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

                # for p in pf.particles:
                #     drawParticle(image, p)
                #     displayParticles = True
                #     while displayParticles:
                #         key = cv2.waitKey(500) & 0xFF
                #         if key == ord('c'):
                #             break
                #         if key == ord('q'):
                #             displayParticles = False
                #     if not displayParticles:
                #         break
                target = algo.target


                #directory =  "out/{}".format(filename)
                #if not os.path.exists(directory):
                #    os.makedirs(directory)
                # cv2.imwrite(
                #     "out/{}/target{}.jpg".format(filename, iterationCount),
                #     utils.cropImage(image, target))

                utils.drawTarget(image, target)
                targetImage = utils.drawParticle(image, target)
                cv2.imshow("particle", targetImage)
                # cv2.imwrite(
                #     "out/{}/image{}.jpg".format(filename, iterationCount),
                #     image)
                # for x in pf.particles:
                    #drawTarget(image, x)
                print "Frame: {0}".format(algo.iterations)
    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        cv2.destroyWindow ("test")
        cv2.destroyWindow ("video")
        cv2.destroyWindow ("particle")
        if algo.pool:
            self.pool.terminate()
            self.pool.join()
