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
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


sys.path.insert(0, 'DARK/python/')
from darknet import *

""" This is simply a script to see how well the eye detection does and how well
it rotate to horizontal a face """


###########################################################################################################


""" Just some auxiliary func """


###########################################################################################################

""" All the nice classes here """

class Yoloeye:

    def __init__(self,thresh=0.2,nms=0.1):
        self.net = load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "DARK/weights/eyes_30k.weights".encode("utf-8"), 0)
        self.meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
        self.thresh = thresh
        self.nms = nms
            
    def detect(self,img):
        R = detect_s(self.net,self.meta,img,thresh=self.thresh,nms=self.nms)
        if len(R) == 0: 
            print('no damn thing detected')
            sys.exit(0)
        return R



###########################################################################################################

""" Recognition script, just takes a filepath and then prints out confidence from what crop """

def horizontalize(filepath,verbose = False):

    if not os.path.exists(filepath): 
        print('File not found')
        sys.exit(0)
    filename = filepath.split('/')[-1]


    #initialization
    yoloeye = Yoloeye(thresh=0.5,nms=0.1)
    img = cv2.imread(filepath)
    detections= yoloeye.detect(img)
    datagen = ImageDataGenerator()


    
    if len(detections) !=2 : 
        print("The picture does not find one face, can't horizontalize")
        sys.exit(0)
    
    box1, box2 = detections[0][2], detections[1][2]
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    x1 , y1 = x1-w1/2,y1-h1/2
    x2 , y2 = x2-w2/2,y2-h2/2
    
    print(x1,y1)
    print(x2,y2)

    m1,m2 = (x1+x2)/2,(y1+y2)/2
    print(m1,m2)
    a = sqrt((y2-m2)**2+(x2-m1)**2)
    b = abs(x2 - m1)
    rot = degrees(acos(b/a))  

    if x2 > m1 and y2 < m2: rot = abs(rot)
    elif x2 > m1 and y2 > m2: rot = -abs(rot)
    elif x2 < m1 and y2 < m2: rot = -abs(rot)
    elif x2 < m1 and y2 > m2: rot = abs(rot)

    print(rot)
    dic = {'theta':rot}
    batch = datagen.apply_transform(img,dic)
    pyplot.imsave('result/'+filename,batch)
    if verbose: 
        img = cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w1),int(y1+h1)),(255,0,0),2)
        img = cv2.rectangle(img,(int(x2),int(y2)),(int(x2+w2),int(y2+h2)),(255,0,0),2)
        cv2.imwrite('verbose/'+filename,img)

###########################################################################################################

        

def main(): 
    #here through syntax, you can either train your file/ do videos_yolos/ or vid_yolo
    array = sys.argv[1:]
    if len(array) >= 3: print('Too many arguments check -h for usage')
    
    if array[0] == '-v':
        if len(array) == 1 : print('Not the right nb of argument, this feature receives only a path/filename')
        if len(array) == 2:  horizontalize(array[1],verbose=True)
    elif array[0] == '-h':
        if len(array) > 1: print('Not the right nb of argument, this requires no extra arguments')
        else: print('-v filepath: to see the pictures with the boxes detection ; filepath: no need for boxes detection')
    elif array[0] != '-v': horizontalize(array[0])
    else:
        print("argument unknown check -h for usage")

if __name__ == '__main__':
    main()