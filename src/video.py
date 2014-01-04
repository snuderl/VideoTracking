import cv2
from algorithm import Algorithm
import cv
import utils
from os import listdir
from os.path import isfile, join
import re
from meanshif import Meanshift
from meanshift_particle import MeanshiftParticleAlgorithm
import numpy as np

#inFile = "../data/Vid_B_cup.avi"
#outFile = "../data/Vid_B_cup_detected_ada.avi"




def process(inFile, outFile, targets, algo):
    capture = cv2.VideoCapture(inFile)
    retval, image = capture.read()
    locations = []
    if retval:
        writer = cv2.VideoWriter(outFile + ".avi", 
            fps=25,
            fourcc=cv2.cv.CV_FOURCC(*"DIVX"),
            frameSize=image.shape[0:2][::-1])
        algorithms = []
        for x in targets:
            algo.start(image, x)
            algorithms.append(algo)
            utils.drawTarget(image, algo.target)
        writer.write(image)

    w,h = image.shape[:2]
    while retval:       
        retval, image = capture.read()
        locations.append(np.array(algo.target) / [h, w, h, w])
        if retval:
            for algo in algorithms:
                algo.next(image)
                color = (255, 0, 0)
                if algo.valid:
                    color = (0, 255, 0)
                utils.drawTarget(image, algo.target, color)
            writer.write(image)

    utils.write(outFile + ".txt", inFile, locations)



path = "../data/"
files = [f for f in listdir(path) if isfile(join(path,f))]
videos = [x for x in files if x.endswith(".avi")]
texts = [x for x in files if x.endswith(".txt")]



for video in videos:
    name = video[:5]
    print "Starting file ", name
    targets = [x for x in texts if x.startswith(name)]

    targetsCoord = []
    for target in targets:
        with open(path+target) as f:
            target = map(float, f.readlines()[1].split()[1:])
            targetsCoord.append((target[0]*320, target[1]*240, target[2]*320, target[3]*240))       

    process(path+video, path+"Tracked/"+name + "meanPF", targetsCoord, MeanshiftParticleAlgorithm())
    process(path+video, path+"Tracked/"+name + "mean", targetsCoord, Meanshift())
    process(path+video, path+"Tracked/"+name + "PF", targetsCoord, Algorithm(1))