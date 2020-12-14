import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

inputDir = "testTrainingSet1\\other\\"
outputDir = "testTrainingSet2\\other\\"

files = [join(inputDir, f) for f in listdir(inputDir) if isfile(join(inputDir, f))]

for path in files :
    filename = os.path.basename(path)
    filename = "".join(filename.split(".")[:-1])
    img = cv2.imread(path)
    if(len(img) == 128 and len(img[0]) == 128):
        continue
    cv2.imwrite(join(outputDir, filename+".jpg"), cv2.resize(img, (128,128)))
    os.remove(path)
    print("Good for "+filename)