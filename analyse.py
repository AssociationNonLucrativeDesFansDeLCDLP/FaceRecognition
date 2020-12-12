import cv2
import numpy as np
from utils import *
from faceDetectors import *
import random

PATH_IN = "test.mp4"
PATH_OUT = "out.mp4"

#--------------------- Program main --------------------------
if __name__ == "__main__":
    print("Counting input frames... ", end='')

    (frameCount,size,fps) = getVideoFramesCountDimFPS(PATH_IN)

    print(f"done : {frameCount} frames, size {size}, {fps} FPS")



    print("Opening files... ", end='')

    vIn = cv2.VideoCapture(PATH_IN)
    vOut = cv2.VideoWriter(PATH_OUT, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    print(f"done")



    print("Face detection initializing... ", end='')

    # values = [10,15,20]
    # colors = generateDistinguishableColors(len(values))
    # detectors = []
    # for i, value in enumerate(values):
    #     detectors.append(SimpleFaceDetector(color=colors[i],scaleFactor=1.05, minNeighbors=value,name=f"v{value}"))

    # detectors = [
    #     SimpleFaceDetector("s0.8m5", color=(255,255,0),scaleFactor=1.03, minNeighbors=3),
    #     SimpleFaceDetector("s0.8m5", color=(255,0,255),scaleFactor=1.03, minNeighbors=8),
    #     SimpleFaceDetector("s0.8m5", color=(255,0,255),scaleFactor=1.03, minNeighbors=11),
    # ] 

    detectors = [
        AddUnifiedFaceDetector("AddUni", detectors=[
            SimpleFaceDetector("", scaleFactor=1.4, minNeighbors=5),
            SimpleFaceDetector("", scaleFactor=1.05, minNeighbors=13),
            SimpleFaceDetector("", scaleFactor=1.2, minNeighbors=7)
        ], color=(255,0,0)),
        InterUnifiedFaceDetector("InterUni", detectors=[
            SimpleFaceDetector("", scaleFactor=1.4, minNeighbors=5),
            SimpleFaceDetector("", scaleFactor=1.05, minNeighbors=13),
            SimpleFaceDetector("", scaleFactor=1.2, minNeighbors=7)
        ], color=(0,255,0)),
    ] 

    print(f"done")

    i=1
    while(vIn.isOpened()):
        try:
            print(f"Work in progress for frame {i}/{frameCount}... ", end='')
            ret, img = vIn.read()
            if ret == False:
                print("done!")
                break
            drawText(img, "Skynet - COUSIN, LE PLUARD", 0, 0)

            print(f"detecting... ", end='')

            for detector in detectors :
                detector.treatImage(img)

            print(f"writing... ", end='')

            vOut.write(img)
            print("")
            i+=1
        except KeyboardInterrupt as e:
            print("")
            break

    for detector in detectors :
        detector.report()

    vOut.release()
    vIn.release()
