import cv2
import numpy as np

PATH_IN='flim.mp4'
PATH_OUT='outExtraction\\'
SKIP_FRAME_STEP=100*25

n = 400*25 # initial frame



vIn = cv2.VideoCapture(PATH_IN)

vIn.set(1,n)#  # Where frame_no is the frame you want

imageCount=0
maxImages=2000
essai=4

faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while(vIn.isOpened()):
    if(imageCount>maxImages):
        print(f'{maxImages} images saved')
        break

    ret, img = vIn.read()
    if n%25 == 0:
        print(f"Frame {n}")
    if ret == False:
        print("done!")
        break

    detectedFaces = faceDetector.detectMultiScale(image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.08, minNeighbors=1)
    if(len(detectedFaces) > 0) :
        print(f"Detected faces, exporting and jumping {SKIP_FRAME_STEP} frames further...")
        cv2.imwrite(f"{PATH_OUT}/{essai}_{imageCount}.jpg", img)
        n+=SKIP_FRAME_STEP
        imageCount+=1
        vIn.set(1,n);
    n+=1