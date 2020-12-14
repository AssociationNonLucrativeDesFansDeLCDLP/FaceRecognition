from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from os import listdir
import numpy as np
import cv2
from os.path import join

class DataSet(object):
    """docstring for DataSet"""
    X_train=None
    y_train=None
    X_test=None
    y_test=None
    X=None
    Y=None
    encoder=None
    transfomed_label=None
    def __init__(self, X=None, Y=None):
        super(DataSet, self).__init__()
        print('instantiating DataSet')
        self.X=X
        self.Y=Y
    
    def split(self):
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.33, random_state=42)
            print('split dataset : Training set : '+str(len(self.y_train))+' instances, Test set : '+str(len(self.y_test))+' instances')
        except:
            print('error when splitting the data')

    def extractImages(self, path, isTrainSet=False, isTestSet=False):
        persons=listdir(path)
        
        try:
            persons.remove('.DS_Store')
        except:
            pass

        encoder=LabelBinarizer()
        transfomed_label=encoder.fit_transform(persons)
        self.encoder=encoder
        self.transfomed_label=transfomed_label
        
        data=[]
        label=[]
        for person in persons:
            faces=listdir(path+person)
            try:
                faces.remove('.DS_Store')
            except:
                pass
            print("number of faces for "+person+" : "+str(len(faces)))
            for face in faces:
                img=cv2.imread(path+join(person,face))
                image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                data.append(image)
                label.append(self.transfomed_label[np.where(self.encoder.classes_==person)].flatten())
        data, label=np.array(data), np.array(label)
        data_reshaped=data.reshape(len(data), len(data[0]), len(data[0][0]), 1)
        if (isTrainSet):
            self.X_train=data_reshaped
            self.y_train=label
        elif (isTestSet):
            self.X_test=data_reshaped
            self.y_test=label    
        else:
            self.X=data_reshaped
            self.Y=label

    def getXTrain(self):
        return self.X_train
    
    def getYTrain(self):
        return self.y_train
            
    def getXTest(self):
        return self.X_test
    
    def getYTest(self):
        return self.y_test
    
    def getX(self):
        return self.X
    
    def getY(self):
        return self.Y

    def getTrainingSet(self):
        return self.X_train, self.y_train
		
    def getTestSet(self):
        return self.X_test, self.y_test

    def getInputShape(self):
        return np.array(self.X_train).shape[1:]

    def getNClasses(self):
        return len(self.encoder.classes_)

    def getTransformedLabels(self):
        return self.transfomed_label

    