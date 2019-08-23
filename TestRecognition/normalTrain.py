#Imported libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
import sys
from os.path import isfile, join
import re
import pickle
import time
from os import listdir
from copy import deepcopy
from math import *

with open('PetName.txt','r') as f:
    full = [n.strip() for n in f.readlines() if n.strip() != '']

class Model:
    def __init__(self,mod=1):
        
        if mod == 1:self.model = cv2.face.LBPHFaceRecognizer_create()
        elif mod == 2 :self.model = cv2.face.FisherFaceRecognizer_create()
        elif mod == 3 :self.model = cv2.face.EigenFaceRecognizer_create()
        self.mod = mod    

        
    def training(self,array): 
        #array is array of tuple(list) having the image and its label
        Training_Data, Labels = [] , [] 

        for label,im in array:
            Training_Data.append(np.asarray( im, dtype=np.uint8))
            Labels.append(label)

        print('Gathering Data to train on !')
        Labels = np.asarray(Labels, dtype=np.int32) #I mean, i guess database won't exceed 256 for now
        face= np.asarray( Training_Data)

        if self.mod == 1:
            try: self.model.train(face,Labels) 
            except: 
                print("could not train")
                sys.exit(0) 
        else: 
            try: self.model.train(face,Labels) 
            except: 
                print("could not train")
                sys.exit(0) 
        print("Model updated sucessefully, Congratulations")
            
        try :self.model.save('models/test.xml')
        except: print('Could not save the model')


if __name__ == "__main__":
    model = Model()
    mainFolder = 'specimen'
    crop_array = []

    for folder in os.listdir(mainFolder):

        if folder in full: label = full.index(folder)
        else: 
            label = len(full) 
            with open('PetName.txt','a') as f: f.write('\n'+folder)
            full.append(folder)

        number = 0
        for file in os.listdir(mainFolder+'/'+folder): 
            img = cv2.imread(mainFolder+'/'+folder+'/'+file,cv2.IMREAD_GRAYSCALE)
            crop_array.append([label,img])

    model.training(crop_array)