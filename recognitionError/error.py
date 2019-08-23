#Imported libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import re
import pickle
import time
from copy import deepcopy
from math import *
from tqdm import tqdm
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.preprocessing import image                  


#########################################################################################################
""" Nice functions """

def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(90, 160))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)



###########################################################################################
""" Nice classes """ 

class LBPHModel:
    def __init__(self,filename):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        try :
            self.model.read('models/'+filename)
        except:
            print("Alright, you must have a starting model to update later on")
            sys.exit(0)
        self.filename = filename

    def prediction(self,img):
        return self.model.predict(img)

class CNNModel:
    def __init__(self,status):
        
        self.model = Sequential()
        self.status = status
        #Inputs and Labels that are pictures names that we then read
        # Inputs in one array of strings, and labels in another array (ordered)


        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(90, 160, 3)))
        self.model.add(Activation('relu'))
        
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(256, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(256, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  
        self.model.add(BatchNormalization())

        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        
        self.model.add(Flatten())
        # Output layer
        self.model.add(Dense(5))
        self.model.add(Activation('sigmoid'))

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        self.model.summary()

        if self.status == 0 :self.model.load_weights('models/body.h5')
        if self.status == 1 :self.model.load_weights('models/face.h5')
        if self.status == 2 :self.model.load_weights('models/haar.h5')
                    
    def predict(self,imgpath):
        arr = path_to_tensor(imgpath)
        return self.model.predict(arr)

class Error:
    # value  is the total error
    # beta is setting for error computation
    # alpha is setting for error computation
    # status is what kind of prediction and ground truth are used  

    def __init__(self,status):
        self.value = None
        self.percentage = None
        self.status = status #[0,1,2,3] [CNNyolobody,CNNyoloface,CNNhaarcascade,LBPHhaarcascade]
        self.threshLPB = 70 #error thresh
        self.threshCNN = 0.5 #conf thresh 


        if self.status in [0,1,2]: self.Model = CNNModel(self.status)
        else: self.Model = LBPHModel('test.xml')

    def resultGen(self,folder,txt):
        #again txt is ground truth
        array1 = []
        array2 = []

        
        for file in os.listdir(folder):
            if self.status not in [0,1,2]:
                img = cv2.imread(folder+'/'+file,cv2.IMREAD_GRAYSCALE)
                name,res = self.Model.predict(img)
                print(res)
                array1.append(name)
            else: 
                res = self.Model.predict(folder+'/'+file)
                array1.append(res.index(max(res)))
        
        with open(txt,'r') as f:
            contents = f.readlines()
            for line in contents: array2.append(int(line.strip()))

        self.value = sum([array1[i]==array2[i] for i in range(len(array1))])/len(array1)


    def result(self,folder):
        #again ground truth full lady unique label
        array1 = []
        carray1 = []
        array2 = []

        for file in os.listdir(folder): 
            if self.status not in [0,1,2]:
                img = cv2.imread(folder+'/'+file,cv2.IMREAD_GRAYSCALE)
                name,res = self.Model.prediction(img)
                if res < self.threshLPB: carray1.append(int(name))
                else: carray1.append(-1)
                array1.append(int(name))
            else: 
                res = self.Model.predict(folder+'/'+file)
                res = list(res[0])
                if max(res) > self.threshCNN: carray1.append(res.index(max(res)))
                else: carray1.append(-1)
                array1.append(res.index(max(res)))

        #alright this is bad, but we'll find a way later xD
        if self.status == 3: array2 = [7] * len(array1)
        if self.status == 2: array2 = [3] * len(array1)
        if self.status == 1: array2 = [4] * len(array1)
        if self.status == 0: array2 = [3] * len(array1)

        self.value = sum([array1[i]==array2[i] for i in range(len(array1))])/len(array1)
        self.percentage = sum([carray1[i]==array2[i] for i in range(len(carray1))])/len(carray1)
####################################################################################################

if __name__ == "__main__":
    e0 = Error(0)
    e0.result('faceLabels/bodyPic')

    e1 = Error(1)
    e1.result('faceLabels/facePic')

    e2 = Error(2)
    e2.result('faceLabels/cascadePic')

    e3 = Error(3)
    e3.result('faceLabels/cascadePic')

    print('This is the accuracy class for CNNBody: '+str(e0.value))
    print('This is the accuracy thresh for CNNBody: '+str(e0.percentage))

    print('This is the accuracy class for CNNFace: '+str(e1.value))
    print('This is the accuracy thresh for CNNFace: '+str(e1.percentage))

    print('This is the accuracy class for CNNhaar: '+str(e2.value))
    print('This is the accuracy thresh for CNNhaar: '+str(e2.percentage))

    print('This is the accuracy class for LPBhaar: '+str(e3.value))
    print('This is the accuracy thresh for LBPhaar: '+str(e3.percentage))