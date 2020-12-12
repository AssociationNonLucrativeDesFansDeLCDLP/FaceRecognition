import cv2
import numpy as np
from utils import *
from abc import ABC, abstractmethod 

RECTANGLE_MARGIN_PX = 2
RECTANGLE_THICKNESS_PX = 1

DETECTION_MERGE_THRESHOLD_DIVIDER = 3

class FaceDetector(ABC) :
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.stats = []
        self.timings = TimingManager()

    def treatImage(self, img):
        faces = self.getFaces(img)
        self.drawFacesBoundaries(img, faces)

    def drawFacesBoundaries(self, img, faces) :
        for face in faces:
            (x,y,w,h) = face
            cv2.rectangle(img, (max([x-RECTANGLE_MARGIN_PX,0]), max([y-RECTANGLE_MARGIN_PX, 0])), (min([x + w+RECTANGLE_MARGIN_PX, len(img[0])-2]), min([y + h+RECTANGLE_MARGIN_PX, len(img)-2])), color=self.color, thickness = RECTANGLE_THICKNESS_PX)
            drawText(img, f"{self.name}", x, y, self.color)

    def report(self):
        print("")
        print("")
        print(f"######### Stats {self.name}: mean {np.mean(self.stats)}, len {len(self.stats)}")
        self.timings.report(f"{self.name}_dtIm")

    @abstractmethod
    def getFaces(self):
        pass

class SimpleFaceDetector(FaceDetector):
    """OpenCV2 CascadeClassifier based face detector"""
    def __init__(self, name, color=(0, 255, 0), path=cv2.data.haarcascades + "haarcascade_frontalface_default.xml", scaleFactor=1.3, minNeighbors=5):
        FaceDetector.__init__(self, name, color)
        self.faceDetector = cv2.CascadeClassifier(path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def getFaces(self, img) :
        self.timings.start(f"{self.name}_dtIm")
        faces = self.faceDetector.detectMultiScale(image=img, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors)
        self.stats.append(len(faces))
        self.timings.stop(f"{self.name}_dtIm")
        print(f"{self.name}:{len(faces)}... ", end='')
        return faces

class AddUnifiedFaceDetector(FaceDetector):
    """Face detector based on multiple OpenCV2 CascadeClassifier"""
    def __init__(self, name, detectors, color=(0, 255, 0)):
        FaceDetector.__init__(self, name, color)
        self.detectors = detectors

    def getFaces(self, img) :
        self.timings.start(f"{self.name}_dtIm")
        totalFaces = []
        for detector in self.detectors :
            for face in detector.getFaces(img):
                totalFaces.append(face)
        self.timings.stop(f"{self.name}_dtIm")

        print(f"{len(totalFaces)} into ", end='')

        self.timings.start(f"{self.name}_merging")
        faces = self.mergeFaces(totalFaces, img)
        self.stats.append(len(faces))
        self.timings.stop(f"{self.name}_merging")

        print(f"{len(faces)} faces... ", end='')

        return faces

    def mergeFaces(self, faces, img) :
        faces = dict(enumerate(faces))
        for i in range(len(faces)):
            if i not in faces :
                continue
            (x,y,w,h) = faces[i]
            for j in list(faces.keys()) : # iterating over copy of ids list where already merged faces has been removed to save computing
                if int(j) != int(i) and self.calcDist(faces[i], faces[j]) <= w/DETECTION_MERGE_THRESHOLD_DIVIDER :
                    faces[i] = merge2Boxes(faces[i], faces[j], img)
                    del faces[j]
        return list(faces.values())

    def calcDist(self, obj1, obj2):
        # obj1: x, y, w, h
        # obj2: x, y, w, h
        obj1_xmin, obj1_ymin, obj1_xmax, obj1_ymax = obj1
        obj1_xmax += obj1_xmin # adding width to xmax (xmax = x + w)
        obj1_ymax += obj1_ymin # adding height to ymax (ymax = y + h)
        obj2_xmin, obj2_ymin, obj2_xmax, obj2_ymax = obj2
        obj2_xmax += obj2_xmin # adding width to xmax (xmax = x + w)
        obj2_ymax += obj2_ymin # adding height to ymax (ymax = y + h)

        x_dist = min(abs(obj1_xmin-obj2_xmin), abs(obj1_xmin-obj2_xmax), abs(obj1_xmax-obj2_xmin), abs(obj1_xmax-obj2_xmax))
        y_dist = min(abs(obj1_ymin-obj2_ymin), abs(obj1_ymin-obj2_ymax), abs(obj1_ymax-obj2_ymin), abs(obj1_ymax-obj2_ymax))

        dist = x_dist + y_dist
        return dist

    def report(self):
        FaceDetector.report(self)
        self.timings.report(f"{self.name}_merging")
        print(f"{self.name} detector sub-detectors reports ======================================\\\\")
        for detector in self.detectors:
            detector.report()
        print(f"{self.name} detector sub-detectors reports ======================================//")

class InterUnifiedFaceDetector(FaceDetector):
    """Face detector based on multiple OpenCV2 CascadeClassifier"""
    def __init__(self, name, detectors, color=(0, 255, 0)):
        FaceDetector.__init__(self, name, color)
        self.detectors = detectors

    def getFaces(self, img) :
        self.timings.start(f"{self.name}_dtIm")
        totalFaces = []
        for detector in self.detectors :
            for face in detector.getFaces(img):
                totalFaces.append(face)
        self.timings.stop(f"{self.name}_dtIm")

        print(f"{len(totalFaces)} into ", end='')

        self.timings.start(f"{self.name}_merging")
        faces = self.mergeFaces(totalFaces, img)
        self.stats.append(len(faces))
        self.timings.stop(f"{self.name}_merging")

        print(f"{len(faces)} faces... ", end='')

        return faces

    def mergeFaces(self, faces, img) :
        faces = dict(enumerate(faces))
        for i in range(len(faces)):
            if i not in faces :
                continue
            (x,y,w,h) = faces[i]
            for j in list(faces.keys()) : # iterating over copy of ids list where already merged faces has been removed to save computing
                if int(j) != int(i) :
                    interVol = self.inter(faces[i], faces[j])
                    (x2,y2,w2,h2) = faces[i]
                    if interVol != None and interVol > (w2*h2)/DETECTION_MERGE_THRESHOLD_DIVIDER:
                        faces[i] = merge2Boxes(faces[i], faces[j], img)
                        del faces[j]
        return list(faces.values())

    def inter(self, a, b):  # returns None if faces don't intersect
        # a: x, y, w, h
        # b: x, y, w, h
        a_xmin, a_ymin, a_xmax, a_ymax = a
        a_xmax += a_xmin # adding width to xmax (xmax = x + w)
        a_ymax += a_ymin # adding height to ymax (ymax = y + h)
        b_xmin, b_ymin, b_xmax, b_ymax = b
        b_xmax += b_xmin # adding width to xmax (xmax = x + w)
        b_ymax += b_ymin # adding height to ymax (ymax = y + h)
        dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
        dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)
        if (dx>=0) and (dy>=0):
            return dx*dy

    def report(self):
        FaceDetector.report(self)
        self.timings.report(f"{self.name}_merging")
        print(f"{self.name} detector sub-detectors reports ======================================\\\\")
        for detector in self.detectors:
            detector.report()
        print(f"{self.name} detector sub-detectors reports ======================================//")