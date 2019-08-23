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


sys.path.insert(0, 'DARK/python/')
from darknet import *

""" This is simply a script to test out the confidence of a recognition model
We also do all the crop head out part in here at the same time, so that we can just give any image
to start with  """

# this array stores the seen pets, we will store this array later in a file and not write it like this
global full
standSize = (150,150) #we take this a standard size to begin with


with open('PetName.txt','r') as f:
    full = [n.strip() for n in f.readlines() if n.strip() != '']


class Model:
    def __init__(self,filename,mod=1):
        
        if mod == 1:self.model = cv2.face.LBPHFaceRecognizer_create()
        elif mod == 2 :self.model = cv2.face.FisherFaceRecognizer_create()
        elif mod == 3 :self.model = cv2.face.EigenFaceRecognizer_create()
        try :
            self.model.read('models/'+filename)
        except:
            print("Alright, you must have a starting model to update later on")
            sys.exit(0)
        self.filename = filename
        self.mod = mod    

    def prediction(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.model.predict(gray)
        
    def training(self,array,label_nb): #Might need to do some resizing here
        Training_Data, Labels = [] , [] 
        for im in array:
            Training_Data.append( np.asarray( im, dtype=np.uint8))
            Labels.append(label_nb)
        print('Gathering Data to train on !')
        Labels = np.asarray(Labels, dtype=np.int32) #I mean, i guess database won't exceed 256 for now
            
        face= np.asarray( Training_Data)
        if self.mod == 1:
            try: self.model.update(face,Labels) 
            except: sys.exit(0) 
        else: 
            try: self.model.train(face,Labels) 
            except: sys.exit(0) 
        print("Model updated sucessefully, Congratulations")
            
        try :self.model.save('models/'+self.filename)
        except: print('Could not save the model')



###########################################################################################################

""" Recognition script, just takes a filepath and then prints out confidence from what crop """

def recognize(filepath,verbose = False):

    if not os.path.exists(filepath): 
        print('File not found')
        sys.exit(0)


    #initialization
    model = Model('test.xml', mod=1)
    img = cv2.imread(filepath)

    tmp = cv2.resize(img,(150,150),interpolation=cv2.INTER_AREA)
    name, neg = model.prediction(tmp)
    print('conf: '+str(neg)+' name: '+str(name))





###########################################################################################################

        

def main(): 
    #here through syntax, you can either train your file/ do videos_yolos/ or vid_yolo
    array = sys.argv[1:]
    if len(array) >= 3: print('Too many arguments check -h for usage')
    
    if array[0] == '-v':
        if len(array) == 1 : print('Not the right nb of argument, this feature receives only a path/filename')
        if len(array) == 2:  recognize(array[1],verbose=True)
    elif array[0] == '-h':
        if len(array) > 1: print('Not the right nb of argument, this requires no extra arguments')
        else: print('-v filepath: to see the cropped pictures ; filepath: no need for cropped pictures')
    elif array[0] != '-v': recognize(array[0])
    else:
        print("argument unknown check -h for usage")

if __name__ == '__main__':
    main()