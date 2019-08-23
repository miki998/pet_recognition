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
sys.path.insert(0, 'python/')
from darknet import *



""" Use an already good weights to then create cropped ones"""

def reuse(path_to_folder):

    #variables initialization
    net_big =  load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "DARK/weights/body_500k.weights".encode("utf-8"), 0)
    meta_big = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
    thresh = 0.3


    for file in tqdm(os.listdir(path_to_folder)):
        if file.split('.')[-1] == 'jpg':
            img = cv2.imread(path_to_folder+'/'+file)
            width = len(img[0])
            height = len(img)
            r = detect_s(net_big,meta_big, img, thresh = thresh)
            if len(r) != 0:
                for l in range(len(r)):
                    _,_,(x,y,w,h) = r[l]
                    x,y,w,h = int(x),int(y),int(w),int(h)
                    X1,Y1 = x-w//2,y-h//2
                    if X1 < 0: X1 = 0
                    if Y1 < 0: Y1 = 0
                    new = img[Y1:Y1+h,X1:X1+w]
                    cv2.imwrite('tmp/'+str(l)+file,new)




#################################################################################################################################################################################



if __name__ == '__main__':
    reuse('cat_dog')
