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
from tkinter import simpledialog
from tkinter import messagebox

#own libs
sys.path.insert(0, 'DARK/python/')
from darknet import *


#NOTE : We will make the damn code more clean another day

""" 
Main Changes:

- Some bar for peopl to label the current picture taken, so that there is a name to the number we assign
- will use more quadrant to a finner elements
- maybe do multiple circle crop catching 

"""

global full
try :
    with open('PetName.txt','r') as f:
        full = [n.strip() for n in f.readlines() if n.strip() != '']
except:
    print('The required file PetName.txt does not exist in your root folder')

predictSize = (150,150)
    

####################################################################################################################################################

""" Useful Functions """ 

def orientation(img,yoloeye,center,verbose=False): #0 -> Right ; 1 -> Top ; 2 -> Left ;  3 -> Bot ; -1 -> no change (inverse input/output)
    X,Y = center 
    r = yoloeye.detect(img)
    if len(r) != 2: 
        print('Either you guys are trying to mess with me or I messed up and dont find a face' )
        return -1, None

    _,_,(x,y,w,h) = r[0]
    _,_,(x1,y1,w1,h1) = r[1]

    midx, midy = (x+x1)/2, (y+y1)/2
    
    #first compute dist, if bigg then see which quadrant
    if sqrt((midx-X)**2 + (midy-Y)**2) < 50: 
        if verbose: return -1, [(x,y,w,h),(x1,y1,w1,h1)]
        else: return -1, None

    tmp = 100 * sqrt(2)

    print(sqrt((midx-X)**2 + (midy-Y)**2))

    # ok, now see which quadrant 
    if verbose:
        if midx < (X - tmp): return 2, [(x,y,w,h),(x1,y1,w1,h1)]
        elif midx > (X + tmp): return 0, [(x,y,w,h),(x1,y1,w1,h1)]
        elif midy < (Y-tmp): return 1, [(x,y,w,h),(x1,y1,w1,h1)]
        elif midy > (Y+tmp): return 3, [(x,y,w,h),(x1,y1,w1,h1)]
    else: 
        if midx < (X - tmp): return 2, None
        elif midx > (X + tmp): return 0, None
        elif midy < (Y-tmp): return 1, None
        elif midy > (Y+tmp): return 3, None


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
        self.filename = filename

    def load(self):
        try :
            self.model.read('models/'+self.filename)
            return 1
        except:
            print("Alright, I guess you're starting a whole new model")
            return 0

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
        self.bool = False
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
 
        self.trained = bool(self.vid.Model.load())
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width * 1.2, height = self.vid.height)
        self.canvas.pack()
 
        # Buttons
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        #self.select = tkinter.Button(window, text='select', command = self.take )
        #self.select.pack()
        #self.select.place(x=1780, y= 720)

        self.delete = tkinter.Button(window, text='erase', command = self.erase )
        self.delete.pack()
        self.delete.place(x=1780, y= 720)


        # Scroll Bar
        self.text = tkinter.Text(window, wrap="none")
        self.vsb = tkinter.Scrollbar(orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.text.pack(fill="x")

        images = os.listdir('sample')
        for i in range(len(os.listdir('sample'))):
            # b = tk.Button(self, text="Button #%s" % i)
            photo = PIL.ImageTk.PhotoImage(PIL.Image.open('sample/'+images[i]))
            #photo = photo.subsample(2)

            b = tkinter.Label(window,image=photo)
            b.image = photo # keep a reference
            # b.pack(side='bottom',fill='x')
            self.text.window_create("end", window=b)
            #text.insert("end", "\n")



        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
 
        self.window.mainloop()

    def erase(self):
        nb = simpledialog.askstring(title="Test", prompt="Which picture you wanna delete, give its number:")
        os.remove('sample/'+os.listdir('sample/')[int(nb)])
        self.vid.crop.remove(self.vid.crop[int(nb)])

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
            if len(self.vid.crop) < 4 and (not self.bool):
                res,arr = orientation(img,self.vid.yoloeye,(640,380),verbose = True)   

                # draw eyes
                if arr != None: 
                    (x,y,w,h),(x1,y1,w1,h1) = arr
                    print(x,y,w,h)
                    frame = cv2.circle(frame,(int(x+390),int(y+130)),(int(w+h//2)),(0,255,0),3)
                    frame = cv2.circle(frame,(int(x1+390),int(y1+130)),(int(w1+h1//2)),(0,255,0),3)

                #draw main circle 
                frame = cv2.circle(frame,(640,380),250,(255,0,0),5)
                for i in self.vid.quadrant: frame = cv2.ellipse(frame,(640,380),(250,250),0,315-i*90,315-(i-1)*90,(0,0,255),5)
                
                if res != -1 or res in self.vid.quadrant:
                    frame = cv2.ellipse(frame,(640,380),(250,250),0,315-res*90,315-(res-1)*90,(0,0,255),5)
                    if not self.trained : arr = compare(img,self.vid.cascadeExt)
                    else : arr = compare(img,self.vid.cascadeExt,self.vid.Model)
                    if arr != None:
                        self.vid.crop.append(arr)
                        self.vid.quadrant.append(res)
                        self.vid.Model.training(self.vid.crop,[self.vid.label_nb]*len(self.vid.crop))
                        self.trained = True
                self.bool = False
            
            else:
                if not self.bool:
                    x = messagebox.askretrycancel('mikiwiki','Nice pictures gotten, do you want to exit or not')
                    if x == 'Yes':
                        nb = simpledialog.askstring(title="mikiwiki", prompt="What is the name of that pet?:")
                        with open('PetName.txt','a') as f :
                            f.write(nb+'\n')
                        sys.exit(0)
                    else: 
                        self.vid.crop.clear()
                        self.vid.quadrant.clear()
            #put it on canvas
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(768, 380, image = self.photo, anchor = tkinter.CENTER)

        self.text.delete('1.0',tkinter.END)
        images = os.listdir('sample')
        for i in range(len(os.listdir('sample'))):
            photo = PIL.ImageTk.PhotoImage(PIL.Image.open('sample/'+images[i]))

            b = tkinter.Label(self.window,image=photo)
            b.image = photo # keep a reference
            self.text.window_create("end", window=b)
            if i > 6: self.text.insert("end", "\n")

        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
       # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)   

        #exterior elements
        self.yoloeye = Yoloeye(thresh=0.5)
        self.Model = LBPHModel('new.xml')
        self.cascadeExt = HaarCascadeExt()
        self.crop = []
        self.quadrant = []
        self.label_nb = len(full)

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


def compare(img,cascadeExt,Model=None): #we shall do detection here too to not put to much task on the app class
    # True/Cropped pic -> the crop is good  ; False/Cropped pic -> the crop is too similar to what we already have
    #img need to be at least the small circle that we put the pet's face in 
    width,height = len(img[0])//2 ,len(img)//2

    r = cascadeExt.detect(img,(width,height))
    if len(r) == 0: return

    new = sorted(r,key = lambda A:A[2]*A[3])
    x,y,w,h = new[-1]
    x,y = int(x),int(y)
    if x < 0 : x = 0
    if y < 0: y = 0
    w,h = int(w), int(h) #trying to extend a bit more space for recognition
    pred2 = cv2.resize(img[y:y+h,x:x+w],predictSize,interpolation=cv2.INTER_CUBIC)

    if Model != None:
        _ , neg = Model.prediction(pred2)
        if neg > 60: 
            cv2.imwrite(str(len(os.listdir('sample'))),pred2)
            return pred2
        else: return None

    else: 
        cv2.imwrite(str(len(os.listdir('sample'))),pred2)
        return pred2



##################################################################################################################################################################



if __name__ == "__main__":

    # Create a window and pass it to the Application object
    App(tkinter.Tk(), "Tkinter and OpenCV")