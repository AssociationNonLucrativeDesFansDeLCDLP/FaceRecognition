from tensorflow.keras import models
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer


class ClassifiedFaceRecognizer(object):
    path='model/'
    model=None
    encoder=None
    persons=['Dataset_Cite/Dominique_resized', 
             'Dataset_Cite/Chantal_resized', 
             'Dataset_Cite/Obama_resized', 
             'Dataset_Cite/Alain_resized', 
             'Dataset_Cite/Miley_resized', 
             'Dataset_Cite/Gerard_resized', 
             'Dataset_Cite/Jean_resized', 
             'Dataset_Cite/Sam_resized']
    
    def __init__(self, name, color=(0, 255, 0)):
        #Model instantiation
        self.loadModel(self.path)
        self.encode()

    def computeProbabilities(self, img, faces):
        probabilities=[]
        for face in faces:
            (x, y, w, h) = face
            cropedImg = img[y:y+h, x:x+w]
            cropedImg = cv2.cvtColor(cropedImg, cv2.COLOR_BGR2GRAY )
            cropedImg = cv2.resize(cropedImg, (100, 100), interpolation = cv2.INTER_AREA)
            cropedImg.resize(100, 100, 1)
            p = self.model.predict(np.array( [cropedImg,] ) )
            index=np.argmax(p)
            #Proba
            #print(p[0][index])
            a=np.zeros(8)
            a[index]=1
            a=np.asarray([a])
            #nom
            #print(self.encoder.inverse_transform(a))
            probabilities.append([self.encoder.inverse_transform(a), p[0][index]])
        return probabilities

    def loadModel(self, file):
        self.model=models.load_model(file)

    def encode(self):
        self.encoder = LabelBinarizer()
        self.encoder.fit_transform(self.persons)   
