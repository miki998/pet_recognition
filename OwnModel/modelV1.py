#libs outside
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

global full
standSize = (150,150) #we take this a standard size to begin with

"""
Main Idea/modifications:
Compared to the previous version

- we will try to implement the weighting as an intra Label feature i.e same weighting system for any pictures of the pet when doing comparisons. (might add more time to inference)
- 

"""


with open('PetName.txt','r') as f:
    full = [n.strip() for n in f.readlines() if n.strip() != '']

#####################################################################################################################

""" Some auxiliary function because could not find them implemented on python ..."""

def countingAll(doubleArray,maxi):
    array = [0] *(maxi+1)
    for i in range(len(doubleArray)):
        for j in range(len(doubleArray[0])):
            array[int(doubleArray[i,j])] = array[int(doubleArray[i,j])] + 1
    total = sum(array)
    for i in range(len(array)):
        array[i] = array[i]/total
    return array


#####################################################################################################################

""" Nice Classes""" 
class Histograms:

    #this class represent a whole picture
    def __init__(self,maxi,gridX = 3,gridY=3):
        self.__maxi = maxi
        self.__gridX = gridX
        self.__gridY = gridY

        self.vector = [[] for _ in range(self.__gridY*self.__gridX)]

    def load(self,lbp):

        w,h = len(lbp[0]),len(lbp)
        unitw, unith = w//self.__gridX + 1, h//self.__gridY + 1
        for i in range(self.__gridX):
            for j in range(self.__gridY):
                cur = lbp[i*unitw:(i+1)*unitw,j*unith:(j+1)*unith]
                self.vector[i*self.__gridY+j] = countingAll(cur,self.__maxi)
    
    def showlbp(self,lbp):
        plt.imshow(lbp,cmap='gray')
        plt.show()

class Model:
    #load: loads from the model arrays to start comparison
    #update: just adds to the txt file model
    #predict: from image compares and take least error label
    #train : from labels array and image array just write it 
    #errorFunc: to compute distance between histograms

    def __init__(self,filename='test.xml',radius=3,METHOD='uniform'):
        self.__radius = radius
        self.__nPoints = 8 * self.__radius
        self.__METHOD = METHOD
        self.__gridX = 9
        self.__gridY = 9
        self.__arrayWeights = []

        self.number = 24
        self.totalCoef = 100
        tmp = [1-i/(self.number + 1) for i in range(self.number+1)]
        total = sum(tmp)

        self.coefs = [self.totalCoef*tmp[i]/total for i in range(self.number+1)]
        self.lbps = [] #index to index match with self.label
        self.label = [] 
        self.filename = filename


    def load(self):
        try:
            with open('baseModel/'+self.filename,'r') as xml:
                current_lbp = []
                contents = xml.readlines()
                for line in contents:
                    if line[2] != ' ':continue
                    cur = line.strip()
                    if cur[:7] == '<label>':
                        label = cur.strip('<label>')
                        self.label.append(int(label))
                        self.lbps.append(np.array(current_lbp))
                        current_lbp = []
                    elif cur[:6] == '<data>' or len(cur) == 0: continue
                    else:
                        values = [float(value) for value in cur.split()] 
                        current_lbp.append(np.array(values))
                    
                print()
            self.lbps = np.array(self.lbps)
            
        except:
           print('damn u idiot, there is no model yet')


    def train(self,images,labels):
        unique = len(set(labels))
        self.__arrayWeights = [[0]*(self.__gridX*self.__gridY) for _ in range(unique)]
        #takes only  gray images
        
        #only saving the file here
        arrays = []
        if len(images) == 0 or len(labels) == 0:
            print('no images')
            sys.exit(0)
        #print(images)
        for img in images:
            img = cv2.resize(img,(150,150),interpolation=cv2.INTER_AREA)
            lbp = local_binary_pattern(img,self.__nPoints,self.__radius,method=self.__METHOD)
            arrays.append(lbp)

        with open('baseModel/'+self.filename,'w') as xml:
            xml.write('<?xml version="1.0"?>\n')
            xml.write('  <radius>'+str(self.__radius)+'<radius>\n')
            xml.write('  <neighbours>'+str(self.__nPoints)+'<neighbours>\n')

            for j in range(len(arrays)):
                array = arrays[j]
                xml.write('    <data>\n')
                for i in range(len(array)):
                    xml.write('    ')
                    for item in array[i]: xml.write(str(item)+' ')
                    if i == len(array)-1: xml.write('\n    <data>\n')
                    else: xml.write('\n')
                xml.write('    <label>'+str(labels[j])+'<label>\n')

        #weight changing
        allLbpErrors = [[] for _ in range(unique)]
        allLbpErrors2 = [dict() for _ in range(unique)]
        for i in range(len(images)):
            for j in range(i+1,len(images)):
                if labels[i] != labels[j]: continue
                oneLbpErrors = self.indexDistance(arrays[i],array[j]) #length self.number
                allLbpErrors[labels[i]].append(oneLbpErrors)
                for idx in oneLbpErrors:
                    if idx not in allLbpErrors2[labels[i]]: allLbpErrors2[labels[i]][idx] = 1
                    else: allLbpErrors2[labels[i]][idx] = allLbpErrors2[labels[i]][idx] + 1 

        for i in range(unique):
            for L in allLbpErrors[i]:
                for coord in allLbpErrors[i][L]:
                    self.__arrayWeights[i][L][coord] = self.__arrayWeights[i][L][coord] +  (self.coefs[allLbpErrors[i][L].index(coord)]/allLbpErrors2[labels[i]][coord])
        

        print("MODEL TRAINED SUCCESFULLY")

    def update(self,images,label):
        arrays = []
        for img in images:
            img = cv2.resize(img,(150,150),interpolation=cv2.INTER_AREA)
            lbp = local_binary_pattern(img,self.__nPoints,self.__radius,method=self.__METHOD)
            arrays.append(lbp)

        with open('baseModel/'+self.filename,'w') as xml:

            for j in range(len(arrays)):
                array = arrays[j]
                xml.write('    <data>\n')
                for i in range(len(array)):
                    for item in array[i]: xml.write(str(item)+' ')
                    if i == len(array)-1: xml.write('    <data>\n')
                    else: xml.write('\n')
                xml.write('    <label>'+str(label)+'<label>\n')
    

    def predict(self,img): #takes only gray
        global full
        img = cv2.resize(img,(150,150),interpolation=cv2.INTER_AREA)
        lbp = local_binary_pattern(img,self.__nPoints,self.__radius,method=self.__METHOD)
        
        noname = []
        for i in range(len(self.lbps)):
            lbp2 = self.lbps[i]
            noname.append(self.errorFunc(lbp,lbp2,self.label[i]))

        print(noname)
        mini = min(noname)
        idx = noname.index(mini)
        print(self.label)
        
        return mini,full[self.label[idx]]
        

    def errorFunc(self,lbp1,lbp2,label):

        h1 = Histograms(self.__nPoints+1,gridX=self.__gridX,gridY=self.__gridY)
        h2 = Histograms(self.__nPoints+1,gridX=self.__gridX,gridY=self.__gridY)

        
        #creating the one hot feature vector
        h1.load(lbp1)
        h2.load(lbp2)
        total = 0
        v = len(h1.vector[0])
        
        for j in range(self.__gridX*self.__gridY):
            total += self.__arrayWeights[label][j]/v*sum([(h1.vector[j][i]-h2.vector[j][i])**2 for i in range(v)])
        return total
        

    def indexDistance(self,lbp1,lbp2):
        #returns huge vector size self.__nPoints + 1 difference of all
        h1 = Histograms(self.__nPoints,gridX=self.__gridX,gridY=self.__gridY)
        h2 = Histograms(self.__nPoints,gridX=self.__gridX,gridY=self.__gridY)
        h1.load(lbp1)
        h2.load(lbp2)

        errors = []
        for i in range(self.__gridX*self.__gridY):
            er = sum([h1.vector[i][j] - h2.vector[i][j] for j in range(self.__nPoints)])
            errors.append(er)
        sort = sorted(errors)

        return [errors.index(sort[i]) for i in range(self.number)]

#####################################################################################################################
