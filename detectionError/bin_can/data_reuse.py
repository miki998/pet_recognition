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
from keras.preprocessing.image import ImageDataGenerator
from math import *
import tqdm

#from my own libraries
sys.path.insert(0, 'DARK/python/')
from darknet import *
from augmentation import *

"""Use not annoted data and good weights to annotate them and couple it with manual boxing to get more annoted data for training """

def reuse(path_to_folder):

    #variables initialization
    net_big =  load_net("DARK/cfg/yolov3-tiny.cfg".encode("utf-8"), "DARK/weights/yolov3-tiny.weights".encode("utf-8"), 0)
    meta_big = load_meta("DARK/cfg/coco.data".encode('utf-8'))
    thresh = 0.3

    os.makedirs('tmp1')
    print('Created a folder to put in the labels')

    for file in tqdm(os.listdir(path_to_folder)):
        if file.split('.')[-1] == 'jpg':
            filename = file[:-4]
            f = open('tmp1/'+filename+'.txt','w')
            img = cv2.imread(path_to_folder+'/'+file)
            width = len(img[0])
            height = len(img)
            r = detect_s(net_big,meta_big, img, thresh = thresh)
            for (R,P,(x,y,w,h)) in r:
                if R == b'cat': clas = 0
                if R == b'dog': clas = 1
                else: continue
                x,y,w,h = int(x),int(y),int(w),int(h)
                X1,Y1 = x-w//2,y-h//2
                newa,newb,newc,newd = convert((width,height),[X1,Y1,w,h])
                f.write(str(clas)+ ' ' +str(newa)+ ' '+str(newb)+' '+str(newc)+' '+str(newd)+'\n')
            f.close()

def reuseCascade(path_to_folder):
    
    SF,N = 1.03,5
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
    

    os.makedirs('tmp1')
    print('Created a folder to put in the labels')

    for file in tqdm(os.listdir(path_to_folder)):
        if file.split('.')[-1] == 'jpg':
            filename = file[:-4]
            f = open('tmp1/'+filename+'.txt','w')
            img = cv2.imread(path_to_folder+'/'+file)
            width = len(img[0])
            height = len(img)
            r = cascade.detectMultiScale(img,scaleFactor=SF,minNeighbors=N)
            for (x,y,w,h) in r:
                clas = 0
                x,y,w,h = int(x),int(y),int(w),int(h)
                newa,newb,newc,newd = convert((width,height),[x,y,w,h])
                f.write(str(clas)+ ' ' +str(newa)+ ' '+str(newb)+' '+str(newc)+' '+str(newd)+'\n')
            f.close()
#################################################################################################################################################################################




if __name__ == '__main__':
    if not os.path.exists('recipient'): 
        print('No recipient folder for data reusage')
        sys.exit(0)
    #reuse('recipient')
    reuseCascade('recipient')