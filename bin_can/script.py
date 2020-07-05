#Imported libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
import sys
from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator
from math import *


#own libs
sys.path.insert(0, 'DARK/python/')
from darknet import *

##############################################################################################################################################################

""" Structures for functions """

""" Models """
class Yoloface:

    def __init__(self,thresh=0.2,nms=0.1):
        self.__net = load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/front_prof_230k.weights".encode("utf-8"), 0)
        self.__meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
        self.__thresh = thresh
        self.__nms = nms
        
    def detect(self,img):
        return detect_s(self.__net,self.__meta,img,thresh=self.__thresh,nms=self.__nms)
    
class Yolobody:
    def __init__(self,thresh=0.3,nms=0.1):
        self.__net =  load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/body_500k.weights".encode("utf-8"), 0)
        self.__meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
        self.__thresh = thresh
        self.__nms = nms

    def detect(self,img):
        return detect_s(self.__net,self.__meta,img,thresh=self.__thresh,nms=self.__nms)

class Yoloeye:
    def __init__(self,thresh=0.3,nms=0.1):
        self.__net =  load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/eye_200k.weights".encode("utf-8"), 0)
        self.__meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
        self.__thresh = thresh
        self.__nms = nms

    def detect(self,img):
        return detect_s(self.__net,self.__meta,img,thresh=self.__thresh,nms=self.__nms)

class HaarCascadeExt:
    def __init__(self,SF=1.03,N=6):
        self.__cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
        self.__SF = SF
        self.__N = N

    def detect(self,img,dim):
        return self.__cat_ext_cascade.detectMultiScale(img,scaleFactor=self.__SF,minNeighbors=self.__N,minSize=dim)
    


###########################################################################################################################################################################################

""" Useful Functions """ 
def horizontalize(img,yoloeye,datagen):

    #initialization
    detections= yoloeye.detect(img)
    
    if len(detections) !=2 : 
        print("The picture does not find one face, can't horizontalize")
        return img
    
    box1, box2 = detections[0][2], detections[1][2]
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    x1 , y1 = x1-w1/2,y1-h1/2
    x2 , y2 = x2-w2/2,y2-h2/2

    m1,m2 = (x1+x2)/2,(y1+y2)/2
    a = sqrt((y2-m2)**2+(x2-m1)**2)
    b = abs(x2 - m1)
    rot = degrees(acos(b/a))  

    if x2 > m1 and y2 < m2: rot = abs(rot)
    elif x2 > m1 and y2 > m2: rot = -abs(rot)
    elif x2 < m1 and y2 < m2: rot = -abs(rot)
    elif x2 < m1 and y2 > m2: rot = abs(rot)

    dic = {'theta':rot}
    batch = datagen.apply_transform(img,dic)

    return batch



def convert_video_toFrame(filename):
    # Model Prep
    yoloface = Yoloface(thresh=0.8)
    yolobody = Yolobody(thresh=0.4)
    yoloeye = Yoloeye(thresh=0.5)
    cascadaExt = HaarCascadeExt(SF=1.03,N=6)

    datagen = ImageDataGenerator()
    number = 0

    cap = cv2.VideoCapture(filename) #cap
    
    while True:
        number += 1
        ret,img = cap.read()
        if not ret: break
        width,height = len(img[0]),len(img)

        r = yolobody.detect(img)   
        if len(r) == 0: continue
        _,_,(x,y,w,h) = r[0]
        x,y,w,h = int(x),int(y),int(w),int(h)
        Xc,Yc = x-w//2,y-h//2
        W,H = w,h
        if Xc < 0: Xc = 0
        if Yc < 0: Yc = 0  
        next = img[Yc:Yc+h//1,Xc:Xc+w//1]
        
        cv2.imwrite('tmp/body/'+str(number)+'.jpg',next)

        r1 = yoloface.detect(next)
        if len(r1) != 0: (R,P,(x,y,w,h)) = r1[0]
        else: continue
        x,y,w,h = int(x),int(y),int(w),int(h)
        X1,Y1 = x-w//2,y-h//2
        if X1 < 0: X1 = 0
        if Y1 < 0: Y1 = 0
        next = next[Y1:Y1+h//1,X1:X1+w//1]
        
        cv2.imwrite('tmp/face/'+str(number)+'.jpg',next)

        next = horizontalize(next,yoloeye,datagen)
        gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        width,height  = int((1/1.8)*len(next[0])),int((1/1.8)*len(next)) #We notice some sort of noise with cascade when pixel image too big (haarcascade training set not large dim)
        catsExt = cascadaExt.detect(gray,(width,height))
        if len(catsExt) == 0: continue
        new = sorted(catsExt,key = lambda A:A[2]*A[3])
        x,y,w,h = new[-1]
        x,y = int(x),int(y)
        if x < 0 : x = 0
        if y < 0: y = 0
        w,h = int(w), int(h) #trying to extend a bit more space for recognition
        
        cv2.imwrite('tmp/haar/'+str(number)+'.jpg',next[y:y+h,x:x+w])


######################################################################################################################################################################################

if __name__ == "__main__":
    convert_video_toFrame('video/vid6high.mp4')