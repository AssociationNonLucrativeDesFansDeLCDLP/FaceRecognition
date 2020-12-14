import sys
import cv2
import hashlib
import numpy as np
import os
from os import listdir
from os.path import isfile, join

def writeHashName(dirPath, img) :
    img = cv2.resize(img, (128,128))
    filename = hashlib.md5(np.ascontiguousarray(img)).hexdigest()
    cv2.imwrite(dirPath+"/"+filename+".jpg", img)

def auto(inputDir, outputDir, faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")):

    files = [join(inputDir, f) for f in listdir(inputDir) if isfile(join(inputDir, f))]

    i, j = 0, 0

    for path in files :
        i += 1
        img = cv2.imread(path)

        if img is not None:
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detectedFaces = faceCascade.detectMultiScale(image=imgGray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in detectedFaces:
                print(f"[File {str(i)}/{str(len(files))}] Face {str(j)} : ({x}x, {y}y) {w}x{h} ; 'a' alain, 'z' chantal, 'e' dominique, 'r' gerard,  anything else to reject..")

                cv2.imshow("face", img[y:y+h,x:x+w])
                key = cv2.waitKey(0)
                if key & 0xFF == ord('a'):   
                    name = "alain"
                elif key & 0xFF == ord('z'):
                    name = "chantal"
                elif key & 0xFF == ord('e'): 
                    name = "dominique"
                elif key & 0xFF == ord('r'):
                    name = "gerard"
                else :
                    continue

                output = join(outputDir, name)
                if not os.path.exists(output):
                    os.makedirs(output)

                # Rotations
                angles = range(-15, 16, 7)
                scales = np.arange(-0.10, 0.21, 0.10)

                for angle in angles :
                    faceCenter = (x+w/2,y+h/2)
                    if angle != 0 :
                        rotMat = cv2.getRotationMatrix2D(faceCenter, angle, 1.0)
                        imgNew = cv2.warpAffine(img, rotMat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
                    else :
                        imgNew = img
                    
                    # Scales
                    mY = len(img)
                    mX = len(img[0])
                    for scale in scales :
                        sX = max(0, int(x-w*scale))
                        sY = max(0, int(y-h*scale))
                        sW = min(mX-sX, int(w*(1+scale*2)))
                        sH = min(mY-sY, int(h*(1+scale*2)))
                        # recrop to square for future resizing
                        if sW > sH :
                            d = sW - sH
                            sX -= d/2
                            sW = sH
                        elif sH > sW :
                            d = sH - sW
                            sY -= d/2
                            sH = sW
                            
                        sW = max(sW, sH)

                        imgCropped = imgNew[sY:sY+sH, sX:sX+sW]
                        print(f"Exporting with scale {scale}x and angle {angle}°")
                        writeHashName(output, imgCropped)

                imgCropped = img[y:y+h, x:x+w]
                j+=1

            #cv2.imshow(f'Faces in {path_to_image}', img)
            #cv2.waitKey(0) 
            #cv2.destroyAllWindows()
        else:
            print(path + " image couldnt be read")

def crop(inputDir, outputDir, faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")):

    files = [join(inputDir, f) for f in listdir(inputDir) if isfile(join(inputDir, f))]

    i, j = 0, 0

    for path in files :
        i += 1
        img = cv2.imread(path)

        if img is not None:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detectedFaces = faceCascade.detectMultiScale(image=image, scaleFactor=1.2, minNeighbors=4)

            for (x, y, w, h) in detectedFaces:
                print(f"[File {str(i)}/{str(len(files))}] Face {str(j)} : ({x}x, {y}y) {w}x{h}")
                imgCropped = img[y:y+h, x:x+w]
                writeHashName(outputDir, imgCropped)
                j+=1

            #cv2.imshow(f'Faces in {path_to_image}', img)
            #cv2.waitKey(0) 
            #cv2.destroyAllWindows()
        else:
            print(path + " image couldnt be read")

def flip(inputDir, outputDir) :
    files = [join(inputDir, f) for f in listdir(inputDir) if isfile(join(inputDir, f))]

    for i, path in enumerate(files) :
        filename = os.path.basename(path)
        filename = "".join(filename.split(".")[:-1])
        img = cv2.imread(path)
        imgFlipped = cv2.flip(img, 1)
        cv2.imwrite(outputDir+"/"+filename+"-flipped.jpg", imgFlipped)
        print(f"[File {str(i+1)}/{str(len(files))}] Flipped to {filename}-flipped.jpg!")

def bright(bright, inputDir, outputDir) :
    bright = float(bright) / 100.0

    files = [join(inputDir, f) for f in listdir(inputDir) if isfile(join(inputDir, f))]

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for i, path in enumerate(files) :
        filename = os.path.basename(path)
        if filename.startswith("."):
            print(f"[File {str(i+1)}/{str(len(files))}] {filename} is hidden file, skipping..")
            continue
        filename = "".join(filename.split(".")[:-1])
        if("bright" in filename) :
            print(f"[File {str(i+1)}/{str(len(files))}] {filename} seems to be brighted image, skipping..")
            continue

        try:
            img = cv2.imread(path)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[...,2]=np.where(hsv[...,2]*bright>255,255,hsv[...,2]*bright)

            imgNew = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(f"{outputDir}/{filename}-bright{bright}.jpg", imgNew)
        except Exception as e:
            print(f"Error while processing image {path}; skipping")
            raise e
            continue
        print(f"[File {str(i+1)}/{str(len(files))}] Brightness +({bright}) to {filename}-bright{bright}.jpg!")

def rotate(angle, inputDir, outputDir):
    angle = float(angle)

    files = [join(inputDir, f) for f in listdir(inputDir) if isfile(join(inputDir, f))]

    for i, path in enumerate(files) :
        filename = os.path.basename(path)
        filename = "".join(filename.split(".")[:-1])
        if("rotate" in filename) :
            print(f"[File {str(i+1)}/{str(len(files))}] {filename} seems to be rotated image, skipping..")
            continue
        img = cv2.imread(path)

        imgCenter = tuple(np.array(img.shape[1::-1]) / 2)
        rotMat = cv2.getRotationMatrix2D(imgCenter, angle, 1.0)
        imgNew = cv2.warpAffine(img, rotMat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        cv2.imwrite(f"{outputDir}/{filename}-rotate{angle}.jpg", imgNew)
        print(f"[File {str(i+1)}/{str(len(files))}] Rotated {angle} deg to {filename}-rotate{angle}.jpg!")

if __name__ == "__main__":
    if(len(sys.argv)) < 2 :
        print("Usage: python dataPreparation.py crop/flip/rotate/bright")
    else :
        mode = sys.argv[1]
        if mode == "crop" :
            if len(sys.argv) < 4 :
                print("crop: Detects faces in every images of inputDir and extract them into outputDir")
                print("Usage: python dataPreparation.py crop inputDir outputDir")
            else :
                crop(sys.argv[2], sys.argv[3])
        elif mode == "flip" :
            if len(sys.argv) < 3 :
                print("flip: Flips honrizontally every image of inputDir into outputDir")
                print("Usage: python dataPreparation.py flip inputDir [outputDir]")
            else :
                flip(sys.argv[2], sys.argv[2] if len(sys.argv) < 4 else sys.argv[3])
        elif mode == "bright" :
            if len(sys.argv) < 4 :
                print("bright: Creates modified brightness version of every image in inputDir by multiplying with brightFactorPercent/100 into outputDir")
                print("Usage: python dataPreparation.py bright brightFactorPercent inputDir [outputDir]")
            else :
                bright(sys.argv[2], sys.argv[3], sys.argv[3] if len(sys.argv) < 5 else sys.argv[4])
        elif mode == "rotate" :
            if len(sys.argv) < 4 :
                print("rotate: Creates rotated version of every image with angle from inputDir to outputDir")
                print("Usage: python dataPreparation.py rotate angle inputDir [outputDir]")
            else :
                rotate(sys.argv[2], sys.argv[3], sys.argv[3] if len(sys.argv) < 5 else sys.argv[4])
        elif mode == "auto" :
            if len(sys.argv) < 4 :
                print("auto: Multicrop images in different scales and rotations")
                print("Usage: python dataPreparation.py auto inputDir outputDir")
            else :
                auto(sys.argv[2], sys.argv[3])
        else :
            print(f"Mode '{sys.argv[1]}' unknown ! Usage: python dataPreparation.py crop/flip/rotate/bright")