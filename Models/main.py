import sys
import numpy as np
from collections import Counter
import Model, DataSet
import ClassifiedFaceRecognizer
import cv2


def hypertrain(dataSet, file):
    print('HyperParameter training')
    dataSet.split()
    n_dense=[128, 312, 512]
    depth=[1, 2]
    count=0
    epochs=5
    print('computing ' + str(len(n_dense)*len(depth))+' models')
    for n in n_dense:
        for d in depth:
            print(str(count+1)+'/'+str(len(n_dense)*len(depth)))
            model=Model.Model(name='n'+str(n)+'d'+str(d))
            model.buildModel(dataSet.getInputShape(), dataSet.getNClasses(), n_dense=n, depth=d)
            model.train(dataSet, epochs=epochs)
            model.evaluate(dataSet)
            model.saveResult(file, writeHeader=count==0)
            count+=1
            del model

def kfoldTrain(dataSet, file):
    print('KFold Training')
    K=5
    n_dense=128
    depth=2
    epochs=100
    model=Model.Model(name='model')
    model.trainKFold(dataSet, epochs, K, n_dense, depth, file)

if __name__ == "__main__":
    try:
        path=sys.argv[1]
    except:
        print('incorret arguments')    

    #dataSet=DataSet.DataSet()
    #dataSet.extractImages(path)
    
    #print('Input shape is : '+str(dataSet.getInputShape()))
    #print('Number of classes is :'+str(dataSet.getNClasses()))

    #file='Kfold.csv'

    ###HYPERPARAMETER TRAINING
    #hypertrain(dataSet, file)
    
    ###KFOLD VALIDATION
    #kfoldTrain(dataSet, file)

    #file='test.csv'
        
    path='model/'
    #model=Model.Model(name='test')
    #dataSet.split()
    #model.buildModel(dataSet.getInputShape(), dataSet.getNClasses(), n_dense=128, depth=2)
    #model.train(dataSet, epochs=150)
    #model.saveModel(path)
    #model.train(dataSet, epochs=5)
    #model.accByClass(dataSet, file=file, writeHeader=True)
    #for i in test:
    #    print(len(i))

    path='photos_test/e.png'

    original_image = cv2.imread(path)

    classifier=ClassifiedFaceRecognizer.ClassifiedFaceRecognizer(name='classifier')
    
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detected_faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.2, minNeighbors=4)

    proba=classifier.computeProbabilities(img=original_image, faces=detected_faces)

    print(proba)

    
    