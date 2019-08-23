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
from cat import *
from os import listdir
from copy import deepcopy


def train_from_video(filename,mod = 1, load = None, name = None): #from video returns a model
    cap = cv2.VideoCapture(filename)
        
    cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface.xml')
    cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
    print('Classifiers ready to be used ! Brace yourself !')

    print('Getting frames from video !')
    crop_array = []
    while(True):
        ret, frame =cap.read()
        if not ret: break 
        im = generate(frame,cat_cascade,cat_ext_cascade)
        if len(im) == 0: continue
        im = max(im, key=len)
        crop_array.append(im)
    model = model_training(crop_array,mod,load, name)
    return model

def train_from_live(window_name,mod=1,load = None, name= None, camera_idx = 0): #from live
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)

    cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface.xml')
    cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')

    crop_array = []
    while(True):
        ret, f = cap.read()
        if not ret: break
        im = generate(f,cat_cascade,cat_ext_cascade)
        if len(im) ==0: continue
        im = max(im,key= len)
        crop_array.append(im)
    model = model_training(crop_array,mod,load,name)
    return model

def crop_data_gather(model,data_path='aux_out/data/',pathOut = 'train_cropped/',name = None): # this is also used for training
    #function to physically create a database of cropped faces
    
    #name is the name of the specimen I wanna recognize
    
    data = data_path[:-1]
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    if name == None:
        name = 'unknown'
    try:
        if not os.path.exists(pathOut+name):
            os.makedirs(pathOut+name)
                
    except OSError:
        print ('Error: Creating directory of data')

    #Uncomment to get rid of the files in train_cropped to avoid taking too much time when train
    # the previous trainings is in any case stored in saved model
    #filelist = [f for f in os.listdir(pathOut)]
    #for f in filelist: os.remove(pathOut+f) 

    for i in range(len(onlyfiles)):
        try:
            im = processImage(model,data,onlyfiles[i],pathOut = None, crop = True, one_crop = True)[0]
        except:
            continue
        destination = pathOut + name + '/' + name + str(i)+'.jpg'
        try:
            cv2.imwrite(destination,im)
        except:
            print('No directory found')


def crop_data(model,data_path='aux_out/data/'): #used for unique training, so it takes only one crop

    data = data_path[:-1]
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    
    gray_crop_data = []
    for i in range(len(onlyfiles)):
        try :
            im = processImage(model,data,onlyfiles[i],pathOut = None, crop = True, one_crop = True)[0]
        except:
            continue
        gray_crop_data.append(im)
    return gray_crop_data


def model_training(crop_array,label_numb= 0,mod = 1, load = None, name = None):
    #load is the name of the file

    #from cropped images, we train our model that we return
    Training_Data, Labels = [] , [] 
    

    
    for im in crop_array:
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) in case input is not gray, but cannot apply on already gray
        
        Training_Data.append( np.asarray( im, dtype=np.uint8))
        Labels.append(label_numb)
    print('Gathering Data to train on !')
    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)
    
    # Initialize facial recognizer Or Load existing
    if mod == 1 : model = cv2.face.LBPHFaceRecognizer_create()
    elif mod == 2 :model = cv2.face.FisherFaceRecognizer_create()
    elif mod == 3 :model = cv2.face.EigenFaceRecognizer_create()
    
    if load != None:
        if os.path.exists('models'): 
            try :
                model.read('models/'+load)
            except:
                print('No model found')
    print('Creation of the model done !')
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
    face, label = np.asarray( Training_Data), np.asarray(Labels)

    # Let's train our model
    if mod == 1: model.update(face,label)
    else: model.train(face,label)
    print("Model trained sucessefully, Congratulations")
    
    #save the model trained with the name 
    if name != None:  
        if not os.path.exists('models'): os.makedirs('models')
        model.save('models/'+name+'.xml')
        
    return model



if __name__ == "__main__":
    # by default we load from total, and train on total
    #train_from_live('Capture for training',mod = 1, load = 'total.xml',name='total')
    pass