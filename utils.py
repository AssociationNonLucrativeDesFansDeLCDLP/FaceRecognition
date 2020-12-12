import cv2
import numpy as np
import time
import random

class TimingManager:
    def __init__(self):
        self.timings = {}
        self.lastTimestamps = {}

    def start(self, id) :
        self.lastTimestamps[id] = int(round(time.time() * 1000))
    
    def stop(self, id) :
        if id in self.lastTimestamps :
            if id not in self.timings :
                self.timings[id] = []
            self.timings[id].append(int(round(time.time() * 1000)) - self.lastTimestamps[id])
            del self.lastTimestamps[id]

    def report(self, id) :
        if id not in self.timings:
            print(f"--- /!\\ No timings found for {id} ---")
        else :
            t = self.timings[id]
            print(f"--- Timings for {id} ---")
            print(f"Count:\t\t{len(t)}")
            print(f"Mean:\t\t{np.mean(t)}ms")
            print(f"Variance:\t{np.var(t)}ms")

def drawText(img, text, x, y, color=(0, 255, 0)) :
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.4
    FONT_THICKNESS = 1
    (txtW, txtH), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)

    cv2.putText(img, text,
        (x, y + txtH),
        FONT,
        FONT_SCALE,
        color,
        FONT_THICKNESS)


def getVideoFramesCountDimFPS(path):
    video = cv2.VideoCapture(path)
    count, fps = 0, 0
    width, height = 0, 0
    try:
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        count = countVideoFrames(video)
    try:
        width  = int(video.get(cv2.CV_CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))
    except:
        if video.isOpened():
            width  = int(video.get(3))
            height = int(video.get(4))

        fps = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    return (count, (width, height), fps)

def countVideoFrames(path):
    count = 0
    while True:
        (grabbed, frame) = video.read()
     
        if not grabbed:
            break
        count += 1
    return count

def merge2Boxes(box1, box2, img):
    # box1: x, y, w, h
    # box2: x, y, w, h
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = max(box1[0]+box1[2], box2[0]+box2[2])-x
    h = max(box1[1]+box1[3], box2[1]+box2[3])-y
    # cv2.imshow("box1", img[box1[1]:box1[1]+box1[3], box1[0]:box1[0]+box1[2]])
    # cv2.imshow("box2", img[box2[1]:box2[1]+box2[3], box2[0]:box2[0]+box2[2]])
    # cv2.imshow("merging", img[y:y+h, x:x+w])
    # cv2.waitKey(1)
    # import pdb; pdb.set_trace()
    return [x, y, w, h]

def generateDistinguishableColors(n): 
  rgb_values = []
  r = int(random.random() * 256) 
  g = int(random.random() * 256) 
  b = int(random.random() * 256) 
  step = 256 / n 
  for _ in range(n): 
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    rgb_values.append((r,g,b)) 
  return rgb_values