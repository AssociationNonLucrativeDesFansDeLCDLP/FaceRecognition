import cv2
import numpy as np
from utils import *
from faceDetectors import *
from faceRecognizers import *
import random

PATH_IN = "test.mp4"
PATH_OUT = "out.mp4"
MAX_FRAME = 0

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


    #--- Single value test ---#
    #
    # values = [10,15,20]
    # colors = generateDistinguishableColors(len(values))
    # detectors = []
    # for i, value in enumerate(values):
    #     detectors.append(CVFaceDetector(color=colors[i],scaleFactor=1.05, minNeighbors=value,name=f"v{value}"))

    #--- Multiple detectors test ---#
    # 
    # detectors = [
    #     InterUnifiedFaceDetector("InterUni", detectors=[
    #         MTCNNFaceDetector("MTCNN", color=(255,0,0)),
    #         CVFaceDetector("coarse", scaleFactor=1.4, minNeighbors=5),
    #         # CVFaceDetector("fine", scaleFactor=1.02, minNeighbors=15),
    #         CVFaceDetector("details", scaleFactor=1.08, minNeighbors=13),
    #         CVFaceDetector("mid", scaleFactor=1.2, minNeighbors=7)
    #     ], color=(0,0,255))
    # ]
    # 
    # colors = generateDistinguishableColors(len(detectors))
    # for i, detector in enumerate(detectors):
    #     detector.setColor(colors[i])

    #--- Unified detectors test ---#
    #
    # detectors = [
    #     # AddUnifiedFaceDetector("AddUni", detectors=[
    #     #     CVFaceDetector("", scaleFactor=1.4, minNeighbors=5),
    #     #     CVFaceDetector("", scaleFactor=1.05, minNeighbors=13),
    #     #     CVFaceDetector("", scaleFactor=1.2, minNeighbors=7)
    #     # ], color=(255,0,0)),
    #     InterUnifiedFaceDetector("InterUni", detectors=[
    #         CVFaceDetector("coarse", scaleFactor=1.4, minNeighbors=5),
    #         CVFaceDetector("fine", scaleFactor=1.02, minNeighbors=15),
    #         CVFaceDetector("details", scaleFactor=1.08, minNeighbors=10),
    #         CVFaceDetector("mid", scaleFactor=1.2, minNeighbors=7)
    #     ], color=(0,255,0)),
    # ] 

    #--- Face recognizer test ---#
    faceDetector = InterUnifiedFaceDetector("InterUni", detectors=[
            MTCNNFaceDetector("MTCNN", color=(255,0,0)),
            CVFaceDetector("coarse", scaleFactor=1.4, minNeighbors=5),
            # CVFaceDetector("fine", scaleFactor=1.02, minNeighbors=15),
            CVFaceDetector("details", scaleFactor=1.08, minNeighbors=13),
            CVFaceDetector("mid", scaleFactor=1.2, minNeighbors=7)
        ], color=(0,0,255))



    detectors = [
        ClassifiedFaceRecognizer(name="MainRecog", faceDetector=faceDetector)
    ]

    print("done")

    i=1
    while(vIn.isOpened()):
        try:
            print(f"Work in progress for frame {i}/{frameCount}... ", end='')
            ret, img = vIn.read()
            if ret == False or (MAX_FRAME > 0 and i >= MAX_FRAME):
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
            print("Work interrupted, breaking..")
            break

    for detector in detectors :
        detector.report()

    vOut.release()
    vIn.release()
