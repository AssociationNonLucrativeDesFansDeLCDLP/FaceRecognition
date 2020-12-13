import cv2
import numpy as np
from utils import *
from abc import ABC, abstractmethod 
import random

RECTANGLE_MARGIN_PX = 2
RECTANGLE_THICKNESS_PX = 1

DETECTION_MERGE_THRESHOLD_DIVIDER = 3

class FaceRecognizer(ABC) :
    def __init__(self, name, color, faceDetector, drawUnknown=True):
        self.name = name
        self.color = color
        self.timings = TimingManager()
        self.faceDetector = faceDetector
        self.drawUnknown = drawUnknown

    def treatImage(self, img):
        self.timings.start(f"{self.name}_faces")
        faces = self.faceDetector.getFaces(img)
        self.timings.stop(f"{self.name}_faces")

        self.timings.start(f"{self.name}_probs")
        probs = self.computeProbabilities(img, faces)
        self.timings.stop(f"{self.name}_probs")

        self.timings.start(f"{self.name}_draw")
        self.drawFacesBoundariesWithProb(img, faces, probs)
        self.timings.stop(f"{self.name}_draw")

    def drawFacesBoundariesWithProb(self, img, faces, probs, color=None, drawUnknown=None) :
        if color == None:
            color = self.color
        if drawUnknown == None:
            drawUnknown = self.drawUnknown
        for i, face in enumerate(faces):
            if probs[i][0] == "Unknown" :
                if drawUnknown :
                    color = (0,0,255)
                else :
                    continue
            (x,y,w,h) = face
            cv2.rectangle(img, (max([x-RECTANGLE_MARGIN_PX,0]), max([y-RECTANGLE_MARGIN_PX, 0])), (min([x + w+RECTANGLE_MARGIN_PX, len(img[0])-2]), min([y + h+RECTANGLE_MARGIN_PX, len(img)-2])), color=color, thickness = RECTANGLE_THICKNESS_PX)
            drawText(img, f"{probs[i][0]} {probs[i][1]}%", x, y, color)

    def report(self):
        print("")
        print("")
        print("")
        print(f"#====================  FaceRecognizer {self.name}  ====================#")
        print("")
        print("")
        print(">>>>>>>>>> Face detector report <<<<<<<<<<")
        self.faceDetector.report()
        print("")
        print("")
        print("")
        print("")
        print("")
        print("")
        print(">>>>>>>>>> Recognition timings report <<<<<<<<<<")
        self.timings.report()

    def setColor(self, color):
        self.color = color

    @abstractmethod
    def computeProbabilities(self, img, faces):
        pass

class ClassifiedFaceRecognizer(FaceRecognizer):
    """Neural network face classifier with TensorFlow"""
    def __init__(self, name, faceDetector, color=(0, 255, 0)):
        FaceRecognizer.__init__(self, name=name, color=color, faceDetector=faceDetector)
        # Model initialisation

        # self.model =



    def computeProbabilities(self, img, faces):
        # Extracts faces (face in faces = (x, y, w, h)) in img and return array of class with probability 
        # self.model.predict...

        # probabilities = [["Unknown", 0.32], ["Alain Chabat", 0.98], ...]
        # return probabilities
        return list(zip([random.choice(["RandomA", "RandomB", "Unknown"]) for i in faces], [float(random.randint(1,100))/100.0 for i in faces]))