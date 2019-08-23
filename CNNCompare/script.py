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

from model import *

if __name__ == '__main__':
    windowName = 'Its OK to be Funny'


    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    Model = Recog()
    Model.model.load_weights('recog.h5')
    
    full = []
    for file in os.listdir('data'): full.append(file)

    while cap.isOpened():
        ret,img = cap.read()
        if not ret: break
        new = cv2.resize(img,(160,90),interpolation=cv2.INTER_AREA)
        new = np.expand_dims(new,axis=0)

        print(full)
        print(Model.model.predict(new))

        cv2.imshow(windowName,img)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break