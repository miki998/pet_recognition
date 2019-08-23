import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.models import load_model
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import time


IMAGE_SIZE = 512

class Cat:
    def __init__(self,dataset):
        # we will be using the following layers CONV-> MAXPOOL 2D -> Flatten -> Dense 128 -> Dense 1 (sigmoid) else relu
        
        self.model = Sequential()

        #Inputs and Labels that are pictures names that we then read
        # Inputs in one array of strings, and labels in another array (ordered)
        self.dataset = dataset 
        self.trained = None
        self.data = [] #where u store the double array version of the pictures
        #self.X_train,self.X_test, self.y_train, self.y_test
        

        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        self.model.add(Activation('relu'))

        #2 Maxpooling layer
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))


        # Maxpooling layer


        self.model.add(Flatten())
        
        self.model.add(Dense(48))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        # Output layer
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

    def processing(self):
        for filename in self.dataset[0]:
            array = cv2.imread('Images/'+filename)
            array = cv2.resize(array,(IMAGE_SIZE,IMAGE_SIZE),interpolation = cv2.INTER_AREA)
            #print(array)
            #array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            self.data.append(array)
        X_train, X_test, y_train, y_test = train_test_split(self.data,self.dataset[1] , 
        test_size=0.1, random_state=42)

        # reshaping to our input layer if needed
        #X_train = X_train.reshape(...)
        #X_test = X_test.reshape(...)

        return X_train, X_test,y_train,y_test

            
    def train(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        X_train,X_test,y_train,y_test = self.processing()
        #print(X_train)
        self.model.fit(np.array(X_train),np.array(y_train), validation_data=(np.array(X_test),y_test),epochs=3)

    def saving(self,name):
        self.model.save(name)


if __name__ == "__main__":
    #fetching
    """
    filenames1 = [f for f in os.listdir('cat_face')]
    #print(filenames1)
    match1 = [1] * len(filenames1)
    filenames2 = [f for f in os.listdir('haar/Negative')]
    #print(filenames2)
    match2 = [0] * len(filenames2)
    filenames = filenames1 + filenames2
    label = match1 + match2
    dataset = filenames,label

    #model making
    classifier = Cat(dataset)
    

    #training
    model = load_model('seventhhmodel.h5')
    classifier.model = model
    classifier.train()
    classifier.saving('eighthmodel.h5')
    #array = cv2.imread('Images/3.png')
    #start= time.time()

    #array = cv2.resize(array,(IMAGE_SIZE,IMAGE_SIZE),interpolation = cv2.INTER_AREA)
    #print(model.predict(np.array([array]))[0][0])
    #print(time.time()-start)"""
    pass