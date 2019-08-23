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


#from my own libraries
#sys.path.insert(0,'libs/')
#from augmentation_recog import * #both of them are kinda optional
#from train import * #both of them are kinda optional
sys.path.insert(0, 'DARK/python/')
from darknet import *

""" This time around, as we have noticed   """

# this array stores the seen pets, we will store this array later in a file and not write it like this
global full
try :
    with open('PetName.txt','r') as f:
        full = [n.strip() for n in f.readlines() if n.strip() != '']
except:
    print('The required file PetName.txt does not exist in your root folder')

predictSize = (150,150)
##############################################################################################################################################################

""" Structures for functions """

""" Models """
class Yoloface:

    def __init__(self,thresh=0.2,nms=0.1):
        self.net = load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/front_prof_230k.weights".encode("utf-8"), 0)
        self.meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
        self.thresh = thresh
        self.nms = nms
        
    def detect(self,img):
        return detect_s(self.net,self.meta,img,thresh=self.thresh,nms=self.nms)
    
    
class Yolobody:
    def __init__(self,thresh=0.3,nms=0.1):
        self.net =  load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/body_500k.weights".encode("utf-8"), 0)
        self.meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
        self.thresh = thresh
        self.nms = nms

    def detect(self,img):
        return detect_s(self.net,self.meta,img,thresh=self.thresh,nms=self.nms)

class HaarCascade:
    def __init__(self,SF=1.03,N=5):
        self.cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface.xml')
        self.SF = SF
        self.N = N
        
    
    def detect(self,img,dim):
        return self.cat_cascade.detectMultiScale(img,scaleFactor=self.SF,minNeighbors=self.N,minSize=dim)

        
class HaarCascadeExt:
    def __init__(self,SF=1.03,N=6):
        self.cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
        self.SF = SF
        self.N = N

    def detect(self,img,dim):
        return self.cat_ext_cascade.detectMultiScale(img,scaleFactor=self.SF,minNeighbors=self.N,minSize=dim)
    
class LBPHModel:
    def __init__(self,filename):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        try :
            self.model.read('models/'+filename)
        except:
            print("Alright, you must have a starting model to update later on")
            sys.exit(0)
        self.filename = filename

    def prediction(self,img):
        return self.model.predict(img)
    
    def training(self,array,label_nb): #Might need to do some resizing here
        Training_Data, Labels = [] , [] 
        for im in array:
            Training_Data.append( np.asarray( im, dtype=np.uint8))
            Labels.append(label_nb)
        print('Gathering Data to train on !')
        Labels = np.asarray(Labels, dtype=np.int32) #I mean, i guess database won't exceed 256 for now
        
        face= np.asarray( Training_Data)
        self.model.update(face,Labels) 
        #except: sys.exit(0) 
        print("Model updated sucessefully, Congratulations")
        
        try :self.model.save('models/'+self.filename)
        except: print('Could not save the model')


""" Storing structures """
class PetSeen:
    #Explanations here
    
    # x,y,w,h -> top left corner coords and dimensions of box
    # t -> time when the pet was detected
    # ID -> its index in the pet list we saw during the current video
    # status -> identity 2: not determined ; 1: known ; 0: unknown
    # name -> its index in the list of names from text file
    # conf -> inverse confidence it is recognized
    # d -> confidence determined by 0: no one yet ; 1: yoloface ; 2:cascade ; 3:cascadeExt 

    def __init__(self,x,y,w,h,t,ID,status,name,conf,d):
        self.x,self.y = x,y
        self.w,self.h = w,h
        self.t = t
        self.ID, self.status = ID,status
        self.name, self.conf= name,conf
        self.d = d

        self.confArray = ConfidenceArray()
class ArrayPets:
    def __init__(self):
        self.array = []
        self.size = 0
        self.previous_size = 0

    def inside(self,coord,dim=None,time=None,ID=None,status=None):
        for pet in self.array:
            if (pet.x,pet.y) != coord: continue
            if dim != None:
                if (pet.w,pet.h) != dim: continue
                if pet.t != time: continue
                if pet.ID != ID: continue
                if pet.status != status: continue
            return True
        return False
    
    def add(self,pet):
        if not self.inside((pet.x,pet.y)):
            self.array.append(pet)
            self.size += 1
        else: 
            tmpPet = self.get((pet.x,pet.y))
            tmpPet.t = pet.t

    def get(self,coords,dim=None):
        for pet in self.array:
            if (pet.x,pet.y) != coords: continue
            if dim != None:
                if (pet.w,pet.h) != dim: continue
            return pet
    
    def update(self,idx,coord,dim,time,ID=None,status=None,name=None,conf=None,d=None):
        self.array[idx].x,self.array[idx].y = coord
        self.array[idx].w,self.array[idx].h = dim
        self.array[idx].t = time
        
        if ID != None:
            self.array[idx].ID, self.array[idx].status = ID, status
            self.array[idx].name, self.array[idx].conf = name, conf
            self.array[idx].d = d
    
    def erase(self,petArray):
        for pet in petArray:
            self.array.remove(pet)
            self.size -= 1


    def show(self):
        print([[(pet.x,pet.y),pet.status,pet.conf]  for pet in self.array])

    def showAll(self):
        print([[(pet.x,pet.y),pet.ID,pet.status,pet.name,int(pet.conf)] for pet in self.array])

class Confidence:
    def __init__(self,ID,conf):
        self.ID = ID
        self.conf = conf


class ConfidenceArray:
    def __init__(self):
        self.array = []
        self.dset = set()
        self.sumCoefs = 0
        self.mean = 0

    def update(self):
        self.sumCoefs = sum([element.ID for element in self.array])
        if self.sumCoefs == 0: self.mean = 0 # just arbitrary values, does not affect much
        else: self.mean = sum(L.ID*L.conf for L in self.array)/self.sumCoefs 
        self.dset = set([element.ID for element in self.array])
    
    def show(self):
        print([(obj.ID,obj.conf) for obj in self.array])
        
            
class Temporary():
    #Explanations here
    def __init__(self):
        self.change_array = []
        self.present = dict()
        self.get_rid = []
        self.still = []
        


""" Verbose related """
class Verbose:
    #Explanations here
    def __init__(self):
        self.yoloface = []
        self.cascade = []
        self.cascadeExt = []










##############################################################################################################################################################


""" Auxiliary function """
def convert_frames_to_video(frame_array,pathOut,fps):
    #frame_array already ordered
    #for sorting the file names properly

    height, width, layers = frame_array[0].shape
    size = (width,height)
    
    try:
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    except:
        print('no video written')

    for i in range(len(frame_array)):
        
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def closest_pair(pt,array,present): #TODO improve for other cases
    #return idx of the closest point in array

    #used in the first association part with previous seen points
    challenger = None
    dist = math.inf
    pt = pt[0],pt[1]
    
    if len(array) == 0: return challenger, dist

    points = [(f[2][0],f[2][1]) for f in array]
    not_take = present.keys() #for two points not to have the same neighbour
    for i in range(len(array)):
        if i in not_take: continue
        X,Y = points[i]
        t = sqrt((X-pt[0])**2 + (Y-pt[1])**2)
        if t < dist:
            dist = t
            challenger = i

    return challenger,dist


def valid_file(filename):

    if filename.split('.')[-1] != 'mp4':
        print('not the right format, this receives mp4 format')
        return False
    if not os.path.exists(filename):
        print('file not found')
        return False
    return True





##############################################################################################################################################################

""" From now are functions that are actually used alone possibly""" 



""" Sligthly different version of video_yolos we take the first few detected faces as training, threshold to take it is still to be determined """

# weights scheme: 0.01 -> conf1 ; 0.3 -> conf2 ; 0.1 -> conf3 ex: would need say 2xconf2 + 3xconf3 + 10xconf1
def video_yolos(filename,fps = 25.0,verbose = False): #TODO: add weights to detection from yoloface
    global full

    # Model Prep
    yoloface = Yoloface(thresh=0.3)
    yolobody = Yolobody(thresh=0.3)
    cascada = HaarCascade(SF=1.03,N=5)
    cascadaExt = HaarCascadeExt(SF=1.03,N=6)
    model = LBPHModel('total.xml')

    # Struct Prep
    lapse = 0.2
    pets = ArrayPets()

    # Useful struct
    frame_array = [] #array to store matrix for convesion to vids
    crop_array = [[] for _ in range(100)] #upper bound the numebr 

    if not valid_file(filename): sys.exit(0) 
    file_name = filename.split('/')[-1]
    cap = cv2.VideoCapture(filename) #cap


    number = 0
    nbOfPet = 0 #index for each pet seen during the video
    while True:
        ret,img = cap.read()
        if not ret: break

        pets.show()

        r = yolobody.detect(img)    
        tmp = Temporary()
        verb = Verbose()
        

        #Matching of old points with new points
        for i in range(pets.size):
            idx, _ = closest_pair((pets.array[i].x,pets.array[i].y),r,tmp.present)
            if idx == None: tmp.still.append(i)
            tmp.present[i] = idx
        

        #Updating the pets array 
        cur = time.time()
        pets.previous_size = pets.size

        for i in range(len(r)):
            x,y,w,h = [int(element) for element in r[i][2]]
            if i not in tmp.present:
                pets.add(PetSeen(x,y,w,h,cur,nbOfPet,2,None,math.inf,0))
                nbOfPet += 1
                tmp.change_array.append(True)
            else: 
                pets.update(tmp.present[i],(x,y),(w,h),cur)
                if pets.array[tmp.present[i]].status == 1: tmp.change_array.append(False)
                else: tmp.change_array.append(True)
        #Still updating the pets array (removing non matched after lapse)
        cur = time.time()
        for i in range(pets.previous_size):
            if tmp.present[i] == None: tmpPet = pets.array[i]
            else: tmpPet = pets.array[tmp.present[i]]
            t = tmpPet.t
            if (i not in tmp.present.values() or i in tmp.still) and cur - t > lapse:
                tmp.get_rid.append(tmpPet) 
        pets.erase(tmp.get_rid)



        #Main Loop
        for i in range(len(r)):
            _,_,(x,y,w,h) = r[i]
            x,y,w,h = int(x),int(y),int(w),int(h)
            tmpPet = pets.get((x,y),(w,h))

            Xc,Yc = x-w//2,y-h//2
            W,H = w,h
            if Xc < 0: Xc = 0
            if Yc < 0: Yc = 0  

            #Sanity check
            if tmp.change_array[i] == False: tmp.change_array[i] = random.random() < 0.05

            if tmp.change_array[i]:
                next = img[Yc:Yc+h//1,Xc:Xc+w//1]
                r1 = yoloface.detect(next)

                if len(r1) != 0: (R,P,(x,y,w,h)) = r1[0]
                else: continue

                x,y,w,h = int(x),int(y),int(w),int(h)
                X1,Y1 = x-w//2,y-h//2
                if X1 < 0: X1 = 0
                if Y1 < 0: Y1 = 0
                next = next[Y1:Y1+h//1,X1:X1+w//1]
                verb.yoloface.append((Xc+X1,Yc+Y1,w,h))

                gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
                width,height  = int((1/2.5)*len(next[0])),int((1/2.5)*len(next)) #We notice some sort of noise with cascade when pixel image too big (haarcascade training set not large dim)
                catsExt = cascadaExt.detect(gray,(width,height))
                cats = cascada.detect(gray,(width,height))

                if len(catsExt) == 0 and len(cats) == 0:

                    #cv2.imwrite('testing/'+str(number)+'.jpg',gray)
                    #number += 1
                    #pred1 = cv2.resize(gray,predictSize,interpolation=cv2.INTER_AREA)
                    #name,neg = model.prediction(pred1)
                    #if tmpPet.status == 0 and neg > 75: crop_array[nbOfPet].append(pred1)
                    #tmpPet.name,tmpPet.conf,tmpPet.d = name, neg, 0.01
                    #print('conf1: '+str(neg))
                    continue

                else:
                    if len(catsExt) != 0:
                        new = sorted(catsExt,key = lambda A:A[2]*A[3])
                        x,y,w,h = new[-1]
                        x,y = int(x),int(y)
                        if x < 0 : x = 0
                        if y < 0: y = 0
                        w,h = int(w), int(h) #trying to extend a bit more space for recognition
                        verb.cascadeExt.append((x+X1+Xc,y+Y1+Yc,w,h))
                        pred2 = cv2.resize(gray[y:y+h,x:x+w],predictSize,interpolation=cv2.INTER_AREA)
                        name, neg = model.prediction(pred2)

                        if neg > 55:
                            cv2.imwrite('testing/'+str(neg)+'.jpg',pred2)
                            number += 1

                        if tmpPet.status == 0 and neg >= 60: crop_array[nbOfPet].append(pred2)
                        tmpPet.name, tmpPet.conf, tmpPet.d = name, neg, 0.0000001
                        print('conf2 :'+str(neg)+', name:'+str(name))
                    """
                    elif len(cats) != 0:
                        new = sorted(cats,key = lambda A:A[2]*A[3])
                        x,y,w,h = new[-1]
                        x,y = int(x),int(y)
                        if x < 0 : x = 0
                        if y < 0: y = 0
                        w,h = int(w), int(h) #trying to extend a bit more space for recognition
                        verb.cascade.append((x+X1+Xc,y+Y1+Yc,w,h))
                        
                        pred3 = cv2.resize(gray[y:y+h,x:x+w],predictSize,interpolation=cv2.INTER_AREA)
                        name, neg = model.prediction(pred3)
                        if neg > 55:
                            cv2.imwrite('testing/'+str(number)+'.jpg',pred3)
                            number += 1
                        if tmpPet.status == 0 and neg >= 60 : crop_array[nbOfPet].append(pred3)
                        tmpPet.name, tmpPet.conf, tmpPet.d = name, neg, 0.1
                        print('conf3: '+str(neg))
                    """

                    if len(crop_array[nbOfPet]) >= 3:
                        tmpPet.status = 1
                        full.append('pet'+str(len(full)-1))
                        model.training(crop_array[nbOfPet],len(full)-1)
                        tmpPet.confArray.array = [] 
                        with open("PetName.txt",'a') as text: text.write('\n'+full[-1])
                        crop_array[nbOfPet] = []
                    
        #print()
        #for L in pets.array:
        #    L.confArray.show()
        #    print('another')
        for tmpPet in pets.array:
            
            Xc,Yc = tmpPet.x-tmpPet.w//2, tmpPet.y-tmpPet.h//2
            if Xc < 0: Xc = 0
            if Yc < 0: Yc = 0

            tmpPet.confArray.update() #computing properties of confArray
            conf, sum_coefs = tmpPet.conf, tmpPet.confArray.sumCoefs
            S,mean  = tmpPet.confArray.dset, tmpPet.confArray.mean
            #print("mean: "+ str(mean))
            if conf < 120 and sum_coefs < 1:
                tmpPet.confArray.array.append(Confidence(tmpPet.d, tmpPet.conf))
            

            if conf < 38 and tmpPet.d != 0.01:
                img = cv2.putText(img,full[name],(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                tmpPet.status = 1

            elif sum_coefs >= 1:
                if mean <= 50 and (0.3 in S or 0.1 in S):
                    img = cv2.putText(img,full[name],(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    tmpPet.status = 1
                else:
                    img = cv2.putText(img,'DK',(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    tmpPet.status = 0

            else: img = cv2.putText(img,'DK',(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            img = cv2.rectangle(img,(Xc,Yc),(Xc+W,Yc+H),(0,255,0),2)
        
        if verbose:
            for (x,y,w,h) in verb.yoloface:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            for (x,y,w,h) in verb.cascadeExt:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            for (x,y,w,h) in verb.cascade:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,0),2)

        frame_array.append(img)
    convert_frames_to_video(frame_array,'video_result/rtestnew.mp4',fps)


######################################################################################################################################################################

def main(): 
    #here through syntax, you can either train your file/ do videos_yolos/ or vid_yolo
    array = sys.argv[1:]
    if len(array) >= 4 or len(array) == 0: 
        print('Not correct number of arguments check -h for usage')
        sys.exit(0)
    if array[0] == '-r':
        if len(array) != 2 and len(array) != 3: print('Not the right nb of argument, this feature receives only a path/filename')
        if len(array) == 2: video_yolos(array[1])
        if len(array) == 3:
            if array[1] == '-v': video_yolos(array[2],verbose =True)
    elif array[0] == '-l':
        if len(array) != 1: print('Not the right nb of argument, this requires no extra arguments')
        else: print('might implement, not very interesting')
    elif array[0] == '-h':
        if len(array) > 1: print('Not the right nb of argument, this requires no extra arguments')
        else: print('For live: -l ; for recognition use: -r ; for helper func: -h')
    else:
        print("argument unknown check -h for usage")
    

######################################################################################################################################################################
if __name__ == '__main__':
    main()