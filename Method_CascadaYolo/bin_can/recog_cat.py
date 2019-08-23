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

# My own lib
from cat import *
from train import *
from feat_align import *
sys.path.insert(0, 'python/')
from darknet import *


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

#################################################################################################################################################################################

""" Here are functions used to output frames with recognized faces """

"""From a video, yields a modified video, with either recognized label or non-recognized label added to it """



def recog_video(filename,model, fps = 25.0): #with extension 

    bl = False #Setting to check whether the picture is  a confident enough
    cap = cv2.VideoCapture(filename)

    #tunable parameters 
    SF = 1.03 #scale factors
    N = 3 #minimum neighors
    arb = 300
    
    
    cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface.xml')
    cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
    print('Classifiers ready to be used ! Brace yourself !')

    frame_number = 0
    print('Getting frames from video !')
    if not os.path.exists('aux_out/changeCrop'): os.makedirs('aux_out/changeCrop')

    #Before sending to pahtIn folder, we will delete everything in the pathOut folder
    filelist = [f for f in os.listdir('aux_out/changeCrop')]
    for f in filelist: os.remove('aux_out/changeCrop/'+f) 
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
            
        im = generate(frame,cat_cascade,cat_ext_cascade,SF,N) #Array of cropped that outputs in a specific order that we will use
        if len(im) == 0: continue
        #To rule out small rectangles
        height, width, channels = frame.shape
        minHeight, minWidth = height//12, width//12
        
        array = [] #Store elements of a tuple: name and a bool that shows pass or not the confidence test
        nb_cat = 1
        for fun in im:
            bl = False
            result = model.predict(fun)
            confidence = int(100*(1-(result[1])/arb))
            #print('For'+ str(nb_cat)+' prediction is : '+str(result[0])+'with conf: '+str(confidence))
            if confidence > 90: bl = True
            tmp = result[0],confidence,bl
            array.append(tmp)
            nb_cat += 1
        frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
        cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
              
        
        cur = 0 # index to do the matching 
        # Warning: Here we make an assumption, which is that cats_ext's elements are ordered the same way as generate's output (which makes sense)

        for (x,y,w,h) in cats_ext:
            if w < minWidth or h < minHeight : continue
            X,Y = x-10, y-10
            name,conf,verf = array[cur]
            if verf: 
                img = cv2.putText(frame,str(name)+'conf:'+str(conf),(X,Y), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            else: 
                img = cv2.putText(frame,'DK',(X,Y), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cur += 1


        #print(img)
        file = str(frame_number)+'.jpg'
        try:
            cv2.imwrite('aux_out/changeCrop/crop'+file,img)
        except:
            print('Could not write the file')
        
    

    print('Almost there, we are now releasing the video')
    #now recombining them to get a video
    #saving the video under the name:= "new" + filename
    filename = filename.split('/')[-1]
    outname = "video/"+"new"+filename
    convert_frames_to_video("aux_out/changeCrop/",outname,fps)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




#################################################################################################################################################################################


"""Same as for video recog, but live one """
""" Note the auxiliary function, defined in cat.py, was a quick idea to try making the video more fluid by spreading the workload (as I happen to not know how to thread/fork on python) """



def live_recog(window_name,model,camera_idx = 0):

    bl = False #Setting to check whether there exists a confident enough picture
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    #tunable parameters 
    SF = 1.03 #scale factors
    N = 3 #minimum neighors
    arb = 300
    cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface.xml')
    cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')

    #height, width, channels = f.shape
    #minHeight, minWidth = height/12, width/12

    frame = None

    print('Getting real-time pictures to process')
    while(True):
        # Capture frame-by-frame
        ret, f = cap.read()

        if not ret: break
        #frame (format):= [cats_ext,array,n,name,conf,verf]
        auxiliary(cap,window_name,frame)
        #We get the cropped faces
        im = generate(f,cat_cascade,cat_ext_cascade, live = (cap,window_name,frame)) #Array of cropped
        if len(im) == 0: continue
        #To rule out small rectangles

        auxiliary(cap,window_name,frame)

        array = [] #Store elements of a tuple: name and a bool that shows pass or not the confidence test
        
        for fun in im:
            bl = False

            auxiliary(cap,window_name,frame)
            result = model.predict(fun)
            confidence = int(100*(1-(result[1])/arb))
            #print('For this cat:'+str(confidence))
            if confidence > 90:
                bl = True
            tmp = result[0],confidence,bl
            array.append(tmp)
        
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    
        auxiliary(cap,window_name)
        #cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
        cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N,minSize = (32,32))

        cur = 0 # index to do the matching 
        # Warning: Here we make an assumption, which is that cats_ext's elements are ordered the same way as generate's output (which makes sense)


        auxiliary(cap,window_name)

        n = len(array)
        for (x,y,w,h) in cats_ext:
            #if w < minWidth or h < minHeight : continue
            X,Y = x-10, y-10
            if cur >= n: break
            name,conf,verf = array[cur]
            if verf: 
                img = cv2.putText(f,str(name)+'conf:'+str(conf),(X,Y), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            else: 
                img = cv2.putText(f,'DK',(X,Y), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            img = cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,0),2)
            cur += 1

        try:
            cv2.imshow(window_name,img)
        except:
            print('Could not show the file')
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break   

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



#################################################################################################################################################################################


""" Same as before but with yolo model detection and haarcascade coupled"""

def recog(path_to_img,model_name=None,verbose = False):
    #job is to recognize one thing only
    model = cv2.face.LBPHFaceRecognizer_create()
    
    try :
        model.read(model_name)
    except:
        print('model not found')

    thresh = 0.1
    N = 3
    SF = 1.05

    cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    net = load_net("cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/front_prof_100k.weights".encode("utf-8"), 0)
    meta = load_meta("cfg/cat-dog-obj.data".encode('utf-8'))
    img = cv2.imread(path_to_img)

    r = detect_s(net,meta, img, thresh = thresh)
    r = [(R,P,(x,y,w,h)) for (R,P,(x,y,w,h)) in r if R == b'dog' or R == b'cat']
    if len(r) != 0: (R,P,(x,y,w,h)) = r[0]
    else: return 
    x,y,w,h = int(x),int(y),int(w),int(h)
    X1,Y1 = x-w//2,y-h//2
    if X1 < 0: X1 = 0
    if Y1 < 0: Y1 = 0
    next = img[Y1:Y1+h//1,X1:X1+w//1]
    if verbose: cv2.imwrite('testing/file1.jpg',next)
    gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    print(model.predict(gray))

    
    #datagen = ImageDataGenerator()
    #x = 2
    #y = 2
    #dic = {'zx':x,'zy':y}
    #next = datagen.apply_transform(next,dic)
    #gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    #if verbose: cv2.imwrite('testing/file2.jpg',next)

    cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N,minSize = (30,30)) 
    if len(cats_ext) == 0: return
    print(cats_ext)
    (x,y,w,h) = cats_ext[0]
    gray =  next[y:y+h,x:x+w]
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if verbose: cv2.imwrite('testing/file3.jpg',gray)
    print(model.predict(gray))


#################################################################################################################################################################################




if __name__=='__main__':
    recog('data/black.jpeg',model_name = 'models/lady.xml',verbose = True)