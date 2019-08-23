#Imported libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image, PIL.ImageTk
import os
import sys
from os.path import isfile, join
import re
import pickle
import time
import random
from os import listdir
from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator
from math import *
import tkinter
 
#own libs
sys.path.insert(0, 'DARK/python/')
from darknet import *


####################################################################################################################################################

""" Useful Functions """ 

def orientation(img,yoloeye,center): #0 -> Right ; 1 -> Top ; 2 -> Left ;  3 -> Bot ; -1 -> no change (inverse input/output)
    X,Y = center 
    r = yoloeye.detect(img)
    if len(r) != 2: 
        print('Either you guys are trying to mess with me or I messed up and dont find a face' )
        return -1

    R,P,(x,y,w,h) = r[0]
    R1,P1,(x1,y1,w1,h1) = r[1]

    midx, midy = (x+x1)/2, (y+y1)/2
    
    #first compute dist, if bigg then see which quadrant
    if sqrt((midx-X)**2 + (midy-Y)**2) < 50: return -1
    tmp = 100 * sqrt(2)

    print(sqrt((midx-X)**2 + (midy-Y)**2))

    # ok, now see which quadrant 
    if midx < (X - tmp): return 2
    elif midx > (X + tmp): return 0
    elif midy < (Y-tmp): return 1
    elif midy > (Y+tmp): return 3

##############################################################################################################################################################



""" Nice classes one """ 

class Yoloeye:
    def __init__(self,thresh=0.3,nms=0.1):
        self.__net =  load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "DARK/weights/eye_200k.weights".encode("utf-8"), 0)
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



##############################################################################################################################################################


""" Main Classes """

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
 
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
 
        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
 
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
 
        self.window.mainloop()
 
    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
 
        if ret:
            cv2.imwrite("tmp/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        img =  frame[130:630,390:890]
        if ret:

            res = orientation(img,self.vid.yoloeye,(640,380))            

            frame = cv2.circle(frame,(640,380),250,(255,0,0),5)

            if res == 3: frame = cv2.ellipse(frame,(640,380),(250,250),0,45,135,(0,0,255),5)
            if res == 2: frame = cv2.ellipse(frame,(640,380),(250,250),0,135,225,(0,0,255),5)
            if res == 1: frame = cv2.ellipse(frame,(640,380),(250,250),0,225,315,(0,0,255),5)
            if res == 0: frame = cv2.ellipse(frame,(640,380),(250,250),0,315,405,(0,0,255),5)

            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)

 
class MyVideoCapture:
    def __init__(self, video_source=0):
       # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)   

        self.yoloeye = Yoloeye(thresh=0.5)


        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
 
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()




##############################################################################################################################################################

""" Main Functions """ 

def compare():
    pass



if __name__ == "__main__":

    # Create a window and pass it to the Application object
    App(tkinter.Tk(), "Tkinter and OpenCV")