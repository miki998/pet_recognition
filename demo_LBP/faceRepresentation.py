
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

sys.path.insert(0, 'DARK/python/')
from darknet import *

###########################################################################################################

""" All the nice classes here """

class Yoloeye:

    def __init__(self,thresh=0.2,nms=0.45):
        self.net = load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "DARK/weights/eye_200k.weights".encode("utf-8"), 0)
        self.meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
        self.thresh = thresh
        self.nms = nms
            
    def detect(self,img):
        R = detect_s(self.net,self.meta,img,thresh=self.thresh,nms=self.nms)
        if len(R) == 0: 
            print('no damn thing detected')
            sys.exit(0)
        return R


class Yoloface:

    def __init__(self,thresh=0.2,nms=0.1):
        self.net = load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "DARK/weights/front_prof_230k.weights".encode("utf-8"), 0)
        self.meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
        self.thresh = thresh
        self.nms = nms
                
    def detect(self,img):
        R = detect_s(self.net,self.meta,img,thresh=self.thresh,nms=self.nms)
        if len(R) == 0: 
            print('no damn thing detected')
            sys.exit(0)
        return R

    def crop(self,img):
        R = self.detect(img)
        if len(R) >2: 
            print('I do not do multiple face representation')
            sys.exit(0)
        _,_,(x,y,w,h) = R[0]
        x,y = int(x-w/2),int(y-h/2)
        if x < 0: x = 0
        if y < 0: y = 0
        w,h = int(w),int(h)
        return img[y:y+h,x:x+w]

class HaarCascadeExt:

    def __init__(self,SF=1.03,N=6):
        self.cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
        self.SF = SF
        self.N = N
    
    def detect(self,img,dim):
        R = self.cat_ext_cascade.detectMultiScale(img,scaleFactor=self.SF,minNeighbors=self.N,minSize=dim)
        return R

    def crop(self,img):
        yoloeye = Yoloeye(thresh = 0.1)
        yoloface = Yoloface(thresh=0.3)

        image = yoloface.crop(img)

        dim = len(image[0])//3,len(image)//3
        image = horizontalize(image,yoloeye)

        tmp = self.detect(image,dim)
        if len(tmp) != 1: 
            print('Not good cascade face found')
            sys.exit(0)
        x,y,w,h = tmp[0]
        tmp = image[y:y+h,x:x+w]
        gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        return  gray

#####################################################################################################################

""" useful functions """ 

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=False, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

def horizontalize(img,yoloeye):

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
    if x1 <0 : x1 = 0
    if y2 <0 : y2 = 0
    if y1 <0 : y1 = 0
    if x2 <0 : x2 = 0

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






####################################################################################################

""" Main function for representing the lbp images """


def main(filepath):
    cascade = HaarCascadeExt(SF=1.03,N=5)
    img = cv2.imread(filepath)
    img = cascade.crop(img)
    img = cv2.resize(img,(150,150),interpolation=cv2.INTER_AREA)

    new = local_binary_pattern(img,n_points,radius,method=METHOD)
    lbp = 256*new/25

    width, height = len(img[0]),len(img)
    div = 3
    unitw, unith = width//3+1, height//3+1

    fig = plt.figure()
    ax1 = fig.add_subplot(4,4,1)
    ax1.imshow(img,cmap='gray')
    ax2 = fig.add_subplot(4,4,2)
    ax2.imshow(lbp,cmap='gray')

    #resized = cv2.resize(img,(400,400),interpolation=cv2.INTER_AREA)
    #new1 = local_binary_pattern(resized,n_points,radius,method=METHOD)

    #ax3 = fig.add_subplot(4,4,15)
    #ax3.imshow(256*new1/25,cmap='gray')
    arrays = []
    for i in range(div):
        for j in range(div):
            arrays.append(new[i:(i+1)*unitw,j:(j+1)*unith])

    for i in range(div**2):
        ax = fig.add_subplot(4,4,3+i)
        
        hist(ax,arrays[i])
    plt.show()


###############################################################################################################


if __name__ == '__main__':
    array = sys.argv[1:]
    if len(array) !=1 :
        print('Not the right argument number')
        sys.exit(0)
    else:
        if not os.path.exists(array[0]): 
            print('path does not exist')
            sys.exit(0)
        else:
            main(array[0])
