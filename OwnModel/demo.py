# Libs
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import time
from keras.preprocessing.image import ImageDataGenerator
from math import *


#own libs
from model import *

global full
standSize = (150,150) #we take this a standard size to begin with


with open('PetName.txt','r') as f:
    full = [n.strip() for n in f.readlines() if n.strip() != '']


#####################################################################################################################

def main(filepath):

    lbph = Model()
    lbph.load()
    testImage = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    start = time.time()
    print(lbph.predict(testImage))
    print(time.time()-start)
if __name__ == "__main__":
    
    array = sys.argv[1:]
    if len(array) == 0: 
        print('Damn this is for creator use, don mess this up you idiot')
        sys.exit(0)
    else: main(array[0])


    """ 
    mainFolder = 'specimen'
    training = []
    labels = []


    for folder in os.listdir(mainFolder):
        if folder in full: label = full.index(folder)
        else: 
            label = len(full) 
            with open('PetName.txt','a') as f: f.write('\n'+folder)
            full.append(folder)

        number = 0
        for image in os.listdir(mainFolder+'/'+folder):
            labels.append(label)
            
            img = cv2.imread(mainFolder+'/'+folder+'/'+image,cv2.IMREAD_GRAYSCALE)
            training.append(img)

    lbph = Model()
    lbph.train(training,labels)
    lbph.load()
    #print(lbph.lbps))
    #print(len(lbph.label))
    
    testImage = cv2.imread('test_images/testing.jpg',cv2.IMREAD_GRAYSCALE)
    print(lbph.predict(testImage))

    """
