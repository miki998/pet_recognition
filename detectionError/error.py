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
from tqdm import tqdm

sys.path.insert(0, 'DARK/python/')
from darknet import *

#########################################################################################################
""" Nice functions """

def closest_point(A,array,used):
    x1,y1 = A[0],A[1]
    challenger = inf
    cur = None
    array = [(f[0],f[1]) for f in array]

    for B in array:
        if B in used: continue
        x2,y2 = B
        mini = sqrt((x2-x1)**2+(y2-y1)**2)
        if mini < challenger:
            challenger = mini
            cur = array.index(B)
    
    return cur,challenger

def convert(size, box):

    x = box[0]/1.0 + (box[2])/2.0
    y = box[1]/1.0 + box[3]/2.0
    x = x/size[0]
    y = y/size[1]

    w = box[2]/size[0]
    h = box[3]/size[1]

    return (x,y,w,h)

def deconvert(img_w,img_h,annbox):
    #x,y is top left corners 
    centx, centy = int(annbox[0] * img_w), int(annbox[1] * img_h)
    w,h = annbox[2] *img_w, annbox[3] * img_h
    x,y = centx-w/2, centy-h/2
    return [int(x),int(y),int(w),int(h)]



###########################################################################################
""" Nice classes """ 

class Error:
    # value  is the total error
    # beta is setting for error computation
    # alpha is setting for error computation
    # status is what kind of prediction and ground truth are used  

    def __init__(self,status):
        self.value = None
        self.percentage = None
        self.threshPercentage = None

        self.thresh = 0.4
        self.beta = 5
        self.alpha = 0.5
        self.status = status #[0,1,2,3] [yolobody,tinybody,yoloface,haarcascade]



    def boxError(self,A,B,W,H):
        # we require B to be ground truth
        # and A to be prediction

        x1,y1,w1,h1 = A
        x2,y2,w2,h2 = B
        X,Y = None,None

        #intersections point
        potential = [(x1,y2),(x1,y2+w2),(x1+h1,y2),(x1+h1,y2+w2),
                    (x2,y1),(x2,y1+w1),(x2+h2,y1),(x2+h2,y1+w1)]

        toremove = []
        for coord in potential:
            x,y = coord
            if  x < x1 or x > x1+w1 or x < x2 or x > x2+w2:
                toremove.append(coord)
            elif y < y1 or y > y1+h1 or y < y2 or y > y2+h2:
                toremove.append(coord) 
        for coord in toremove: potential.remove(coord) 
        

        for i in range(len(potential)):
            for j in range(i+1,len(potential)):
                x1,y1 = potential[i]
                x2,y2 = potential[j]
                if x1 == x2: Y = abs(y1-y2) 
                if y1 == y2: X = abs(x1-x2)
         

        if X == None or Y == None:
            integ1 = w1*h1
            integ2 = w2*h2
        else:
            integ1 = w1*h1 - X*Y
            integ2 = w2*h2 - X*Y


        E = 1/(W*H)*self.alpha * integ1 + 1/(W*H)*self.beta * integ2
        return E



    def compare(self,txt1,txt2):
        #again we require txt2 to be ground truth

        filename = txt1.split('/')[-1][:-4]

        img = cv2.imread('recipient/'+filename+'.jpg')
        W,H = len(img[0]),len(img)

        #fetching
        arrayCoord1 = []
        arrayCoord2 = []
        used = []
        match = dict()
        E = 0
        percentageArray = []
        threshArray = []

        with open(txt1,'r') as f:
            contents = f.readlines()
            for line in contents:
                x,y,w,h = [float(l) for l in line.split()[1:]]
                x,y,w,h = deconvert(W,H,(x,y,w,h))
                arrayCoord1.append((x,y,w,h))

        with open(txt2,'r') as f:
            contents = f.readlines()
            for line in contents:
                x,y,w,h = [float(l) for l in line.split()[1:]]
                x,y,w,h = deconvert(W,H,(x,y,w,h))
                arrayCoord2.append((x,y,w,h))
        
        #matching of Coord1 to Coord2
        for value in arrayCoord1:
            idx,closest = closest_point(value,arrayCoord2,used) #it's fine
            if closest == inf: match[value] = None
            else: 
                match[value] = arrayCoord2[idx] 
                used.append(closest)
            

        for value in arrayCoord1:
            if match[value] != None:
                Ep = self.boxError(value,match[value],W,H)
                E += Ep
                percentageArray.append(1-Ep/(self.alpha+self.beta))
                threshArray.append((1-Ep/(self.alpha+self.beta)>=self.thresh))
            else:
                E += self.alpha
                percentageArray.append(self.beta/(self.alpha+self.beta))
                threshArray.append((self.beta/(self.alpha+self.beta)>=self.thresh))
        if len(percentageArray) != 0: return E, sum(percentageArray)/len(percentageArray), sum(threshArray)/len(threshArray)
        else: 
            if  len(arrayCoord2) != 0: return E,0,0
            else: return E,1,1

    def result(self,folderpath):

        #just put in the folderpath of text files prediction that you wanna check here
        totalE = 0
        totalP = 0
        totalT = 0

        if len(os.listdir(folderpath)) != 0:
            for file in tqdm(os.listdir(folderpath)):

                if self.status in [0,1]: E,P,T = self.compare(folderpath+'/'+file,'bodyLabels/groundT/'+file)
                else: E,P,T = self.compare(folderpath+'/'+file,'faceLabels/groundT/'+file)

                totalE += E
                totalP += P
                totalT += T

            self.value = totalE/len(os.listdir(folderpath))
            self.percentage = totalP/len(os.listdir(folderpath))
            self.threshPercentage = totalT/len(os.listdir(folderpath))
        else:
            self.value = 1
            self.percentage = 1 
            self.threshPercentage = 1

####################################################################################################

if __name__ == "__main__":
    e = Error(1)
    e.result('bodyLabels/predictionElse/')
    print('Test with Tiny: '+str(e.value)+'  and  '+str(e.percentage)+'  and  '+str(e.threshPercentage))

    e = Error(0)
    e.result('bodyLabels/predictionOwn/')
    print('Test with Yolobody: '+str(e.value)+'  and  '+str(e.percentage)+'  and  '+str(e.threshPercentage))

    e = Error(2)
    e.result('faceLabels/predictionYolo/')
    print('Test with Yoloface: '+str(e.value)+'  and  '+str(e.percentage)+'  and  '+str(e.threshPercentage))

    e = Error(3)
    e.result('faceLabels/predictionHaar/')
    print('Test with Haar: '+str(e.value)+'  and  '+str(e.percentage)+'  and  '+str(e.threshPercentage))