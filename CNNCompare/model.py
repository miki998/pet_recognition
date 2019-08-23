import sys
import os
import numpy as np
import matplotlib.pyplot as plt


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
from keras.preprocessing import image                  
from tqdm import tqdm





def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(90, 160))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):

    stack = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(stack)

class Recog:
    def __init__(self,dataset=None):

        self.model = Sequential()

        #Inputs and Labels that are pictures names that we then read
        # Inputs in one array of strings, and labels in another array (ordered)
        self.dataset = dataset 
        self.trained = None
        self.data = [] #where u store the double array version of the pictures

        
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

    def processing(self):
        
        self.data = paths_to_tensor(self.dataset[0])
        X_train, X_test, y_train, y_test = train_test_split(self.data,self.dataset[1] ,test_size=0.1, random_state=40)
        print(y_train,y_test)
        return X_train, X_test,y_train,y_test

            
    def train(self):

        X_train,X_test,y_train,y_test = self.processing()
        #print(X_train)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)


        self.model.fit(np.array(X_train),np.array(y_train), validation_data=(np.array(X_test),y_test),epochs=50    )

    def saving(self,name):
        self.model.save(name)


if __name__ == "__main__":
    #fetching
    labels = []
    arrayA = []
    arrayB = []

    for folder in os.listdir('data'):

        labels.append(folder)
        for pic in os.listdir('data/'+folder+'/'):
            arrayB.append(labels.index(folder))
            arrayA.append('data/'+folder+'/'+pic)

    with open('Names.txt','w') as f:
        for name in labels: f.write(name+'\n')
 
    dataset = (arrayA,arrayB)
    mod = Recog(dataset)
    mod.train()
    mod.saving('models/haar.h5')
    
