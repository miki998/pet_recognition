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
sys.path.insert(0,'libs/')
from augmentation_recog import *
from train import *
sys.path.insert(0, 'python/')
from darknet import *

""" In this version, we updates the models, and keep track of each pet by either recognized, not determined yet, or not recognized and will try if good enough crops add it to 
repertoire with a default name, you can modify it by yourself in the txt file. We have a demo of it in report2. """

# this array stores the seen pets, we will store this array later in a file and not write it like this
global full

with open('PetName.txt','r') as f:
    full = [n.strip() for n in f.readlines() if n.strip() != '']



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
    not_take = [f[0] for f in present] #for two points not to have the same neighbour
    for i in range(len(array)):
        if i in not_take: continue
        X,Y = points[i]
        t = sqrt((X-pt[0])**2 + (Y-pt[1])**2)
        if t < dist:
            dist = t
            challenger = i

    return challenger,dist



##############################################################################################################################################################



""" From now are functions that are actually used alone possibly""" 

def vid_yolo(filename,fps = 25.0):
    # this function is used to test how efficient body yolov is efficient
    net=  load_net("cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/body_500k.weights".encode("utf-8"), 0)
    meta = load_meta("cfg/cat-dog-obj.data".encode('utf-8'))
    frame_array = []
    
    name = filename.split('/')[-1]
     
    cap = cv2.VideoCapture(filename)
    
    if not os.path.exists(filename): 
        print('file not found')
        return 
    if filename.split('.')[-1] != 'mp4':
        print('not the right format, this receives mp4 format')
        return 

    print('Classifiers ready to be used ! Brace yourself !')
    thresh = 0.5
    print('Getting frames from video !')
    
    detected_counter = [0,0]
    while (True):
        ret, frame = cap.read()
        
        if not ret:
            break
        detected_counter[1] = detected_counter[1] + 1

        r = detect_s(net,meta, frame, thresh = thresh, nms = 0.1)
        if len(r) != 0: detected_counter[0] = detected_counter[0] + 1
        for (R,P,(x,y,w,h)) in r:
            #if R != b'dog' and R != b'cat': continue
            x,y,w,h = int(x),int(y),int(w),int(h)
            frame = cv2.rectangle(frame,(x-w//2,y-h//2),(x+w//2,y+h//2),(255,0,0),3)
        frame_array.append(frame)
    convert_frames_to_video(frame_array,'video_result/m'+name,fps)
    print("Some sort of test of accuracy for a single pet checking: "+str(detected_counter[0]/detected_counter[1]))



###################################################################################################################################################################



""" Sligthly different version of video_yolos we take the first few detected faces as training, threshold to take it is still to be determined """

# weights scheme: 0.01 -> conf1 ; 0.3 -> conf2 ; 0.1 -> conf3 ex: would need say 2xconf2 + 3xconf3 + 10xconf1
def video_yolos(filename,fps = 25.0,verbose = False): 
    global full
    #Prep
    net = load_net("cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/front_prof_230k.weights".encode("utf-8"), 0)
    meta = load_meta("cfg/cat-dog-obj.data".encode('utf-8'))
    net_big =  load_net("cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/body_500k.weights".encode("utf-8"), 0)
    meta_big = load_meta("cfg/cat-dog-obj.data".encode('utf-8'))
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('models/total.xml')
    cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
    cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface.xml')

    # settings
    thresh1, thresh2 = 0.3,0.2 
    SF,N,SF1,N1 =1.03, 5,1.03,3
    lapse = 0.3 #in seconds

    #array to store matrix for convesion to vids
    frame_array = []

    #keep track of the items, not very expensive since I only store little numbers of them
    seen = [] #is the array for the detected objects its format is (coords,dimensions,id,and status); status -> 0: not known,1: know, 2:not determined yet
    tmp = dict() #used for remmebeing the matching between the coords and the name and confidence
    confidence_array = [[] for _ in range(100)] #confidence of pictures processed before deciding whether we recognize the pet
    crop_array = [[] for _ in range(100)] #for training purpose

    file_name = filename.split('/')[-1]

    #cap
    cap = cv2.VideoCapture(filename)
    if filename.split('.')[-1] != 'mp4':
        print('not the right format, this receives mp4 format')
        return 
    if not os.path.exists(filename):
        print('file not found')
        return 

    number = 0
    pet_number = 0 #index for each pet seen during the video
    while True:

        ret, img = cap.read()
        if not ret:
            break
        #width = len(img[0])
        #height = len(img)


        #step 1 first yolo for pet body
        #But we don't have to get rid of the previous rectangle every time we don't detect, we allow a 0.5 sec span of time, and if during this time detect, then update else after
        #this span, get rid of it therefore, to do that we will need to store the time of each detect
        r = detect_s(net_big,meta_big, img, thresh = thresh1, nms = 0.1 )
        r = [(R,P,(x,y,w,h)) for (R,P,(x,y,w,h)) in r if R == b'dog' or R == b'cat']

        #another set of settings
        #change is used for not computing all the time recognition
        change_array = []
        present = [] #used for storing the matching indexes
        yolo_face = [] #used for verbose
        cascade = []
        cascade2 = []
        get_rid = []
        still = []
        #maxi_dist =  (width*0.5 + height *0.5) * 0.30 #30 percent, so for fast moving things, we just do not really recognize well
        var = len(seen)
        #closest points policy (for now)
        # we associate what each current detected center to the previous detected ones, plus we assume that every seen has a match
        
        for i in range(len(seen)):
            idx, dist = closest_pair(seen[i],r,present)
            #if dist >  maxi_dist : #deals with the case when a pet leaves and a pet comes in at the same time, so that we consider the new pet to not be the same as the one that left
            #    print('not reasonable')
            #    continue
            if idx == None: still.append(i)
            present.append((idx,i)) # present stores both the index from array r (where you have the new points) and the index of the matching point to modify in seen array
        
        present1 = [f[0] for f in present] # list of the new indexes in r
        present2 = [f[1] for f in present] # list of the old indexes in previous seen


        cur = time.time()
        for i in range(len(r)):
            if i not in present1: #there is a new pet recognized that appeared
                x,y,w,h = r[i][2]
                x,y,w,h = int(x),int(y),int(w),int(h)
                if [x,y,w,h,cur] not in [s[:5] for s in seen]: 
                    seen.append([x,y,w,h,cur,pet_number,2]) #append to seen x,y,w,h, note when appending we do modify the seen array BUT do not change the order so the matching is still valid
                    pet_number += 1
                tmp[(x,y,w,h)] = (None,math.inf,0)
                change_array.append(True) #as we have detected a new pet, then need to try recognizing him
            else:
                idx = present1.index(i) #this is the index in seen array of the old point to replace
                x,y,w,h = r[i][2]
                x,y,w,h = int(x),int(y),int(w),int(h)
                a,b,c,d,t,pet_nb,recog = seen[present2[idx]]
                if (a,b,c,d) in tmp:
                    name,conf,fcon = tmp[(a,b,c,d)] #make a copy of it before deleting
                    del tmp[(a,b,c,d)] #delete the old key
                    tmp[(x,y,w,h)] = (name,conf,fcon) #   name and conf defined in the previous loop
                seen[present2[idx]] = [x,y,w,h,cur,seen[present2[idx]][5],seen[present2[idx]][6]]
                
                if seen[present2[idx]][6] == 1: change_array.append(False)
                else: change_array.append(True)

        cur = time.time()
        for i in range(var):
            if (i not in present2 or i in still) and cur - seen[i][4] > lapse: get_rid.append(seen[i])
        for X in get_rid: seen.remove(X)
        #print(still)
        #print(pet_number)
        count = 0
        temp_array = [f[:4] for f in seen]
        for (R,P,(x,y,w,h)) in r: #We iterate over what body we see
            x,y,w,h = int(x),int(y),int(w),int(h)
            x1,y1,w1,h1 = x,y,w,h

            
            idx = temp_array.index([x,y,w,h])
            a,b,c,d,t,pet_nb,recog = seen[idx]    
            #print(seen[idx])
            
            #uncentered variables    
            Xc,Yc = x-w//2,y-h//2
            W,H = w,h
            if Xc < 0: Xc = 0
            if Yc < 0: Yc = 0       
            
            # bool to check whether we need to update or not
            if change_array[count] == False:
                #for sanity check, in case I recognized something but turns out to be wrong, we don't want the label to stay unchanged forever
                if random.random() < 0.05: change_array[count] = True

            if change_array[count]:

                next = img[Yc:Yc+h//1,Xc:Xc+w//1]

                #step2 second yolo only for pet face
                r1 = detect_s(net,meta, next, thresh2)
                #take biggest for now, as we assume the body check gives us separate pets
                if len(r1) != 0: (R,P,(x,y,w,h)) = r1[0]
                else: continue
                x,y,w,h = int(x),int(y),int(w),int(h)
                X1,Y1 = x-w//2,y-h//2
                if X1 < 0: X1 = 0
                if Y1 < 0: Y1 = 0
                next = next[Y1:Y1+h//1,X1:X1+w//1]
                yolo_face.append((Xc+X1,Yc+Y1,w,h))
                
                #step 4 detect with haarcascade
                gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
                
                width,height  = int((1/2.5)*len(next[0])),int((1/2.5)*len(next)) #We notice some sort of noise with cascade when pixel image too big (haarcascade training set not large dim)
                # we abort the recognition of any pic that has pixel reso of lower than (width,height)
                cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N,minSize = (width,height)) 
                cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF1, minNeighbors=N1,minSize = (width,height))        

                if len(cats_ext) == 0 and len(cats) == 0: 
                    name, neg = model.predict(gray)
                    if recog == 0 and neg > 75: crop_array[pet_nb].append(gray)
                    tmp[(x1,y1,w1,h1)] = name,neg,0.01
                    print('conf1 '+str(neg))
                    number += 1
                else: 
                    if len(cats_ext) != 0:
                        new = sorted(cats_ext,key = lambda A:A[2]*A[3])
                        x,y,w,h = new[-1]
                        x,y = int(x-w*0.1),int(y-h*0.1)
                        if x < 0 : x = 0
                        if y < 0: y = 0
                        w,h = int(1.2*w), int(1.2*h) #trying to extend a bit mroe the rect to see if that helps

                        cascade.append((x+X1+Xc,y+Y1+Yc,w,h))
                        name, neg = model.predict(gray[y:y+h,x:x+w])
                        if recog == 0 and neg >= 60 : crop_array[pet_nb].append(gray[y:y+h,x:x+w])
                        
                        tmp[(x1,y1,w1,h1)] = name,neg,0.3
                        print('conf2 '+str(neg))
                        number += 1
                    elif len(cats) != 0:
                        new = sorted(cats,key = lambda A:A[2]*A[3])
                        x,y,w,h = new[-1]
                        x,y = int(x-w*0.1),int(y-h*0.1)
                        if x < 0 : x = 0
                        if y < 0: y = 0
                        w,h = int(1.2*w), int(1.2*h) #trying to extend a bit mroe the rect to see if that helps

                        cascade2.append((x+X1+Xc,y+Y1+Yc,w,h))
                        

                        name, neg = model.predict(gray[y:y+h,x:x+w])

                        if recog == 0 and neg >= 60 : crop_array[pet_nb].append(gray[y:y+h,x:x+w])
                        
                        #cv2.imwrite('testing/'+str(number)+'file.jpg',gray[y:y+h,x:x+w])
                        tmp[(x1,y1,w1,h1)] = name,neg,0.1
                        print('conf3 '+str(neg))
                        number += 1
                    
                #ALright we have enough crops for this specific pet we can now train on it and add it in our repertoire
                if len(crop_array[pet_nb]) >= 3: 
                    seen[idx][6] = 1
                    full.append('pet'+str(len(full)-1))
                    
                    model = model_training(crop_array[pet_nb],label_numb=len(full)-1,mod=1,load= 'total.xml',name = 'total')
                    with open("PetName.txt",'a') as text: 
                        text.write('\n'+full[-1]) 
                    crop_array[pet_nb] = []
            count += 1

        #print(seen)
        #print(full)
        for index in range(len(seen)):
            obj = seen[index]
            x,y,w,h,t,nb_pet,recog = obj

            
            Xc,Yc = x-w//2,y-h//2
            W,H = w,h
            if Xc < 0: Xc = 0
            if Yc < 0: Yc = 0
            

            if (x,y,w,h) in tmp: 
                name,conf,fcon = tmp[(x,y,w,h)]
                coefs = [element[1] for element in confidence_array[nb_pet]]
                sum_coeff = sum(coefs)
                if sum_coeff == 0: mean = 0
                else: mean = sum([element[0]*element[1] for element in confidence_array[nb_pet]])/sum_coeff
                print(sum_coeff,mean)
                #print(name,conf,fcon)
                if conf < 120 :
                    if conf not in [element[0] for element in confidence_array[nb_pet]]:
                        if sum_coeff < 1: 
                            confidence_array[nb_pet].append([conf,fcon]) #weird values are not taken
                

                if conf < 38 and fcon != 0.01: 
                    #print(name)
                    img = cv2.putText(img,full[name],(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                elif sum_coeff >= 1:

                    if mean <= 50 and (0.3 in coefs or 0.1 in coefs):
                        img = cv2.putText(img,full[name],(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                        seen[index][6] = 1 #known pet
                    else: 
                        seen[index][6] = 0 #unknown pet so we will just give it a label anyways
                        confidence_array[nb_pet] = []
                        
                else: img = cv2.putText(img,'DK',(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            img = cv2.rectangle(img,(Xc,Yc),(Xc+W,Yc+H),(0,255,0),2)

        if verbose:
            for (x,y,w,h) in yolo_face:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            for (x,y,w,h) in cascade:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            for (x,y,w,h) in cascade2:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,0),2)
        frame_array.append(img)
               
    #print(file_name)
    convert_frames_to_video(frame_array,'video_result/r'+file_name,fps)




##############################################################################################################################################################


"""Just live/webcam version of it """

def live_yolo_recog(window_name,camera_idx = 0,verbose = False):
    global full
    #Prep
    net = load_net("cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/front_prof_230k.weights".encode("utf-8"), 0)
    meta = load_meta("cfg/cat-dog-obj.data".encode('utf-8'))
    net_big =  load_net("cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "weights/body_500k.weights".encode("utf-8"), 0)
    meta_big = load_meta("cfg/cat-dog-obj.data".encode('utf-8'))
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('models/total.xml')
    cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
    cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface.xml')

    # settings
    thresh1, thresh2 = 0.3,0.2 
    SF,N,SF1,N1 =1.03, 5,1.03,3
    lapse = 0.3 #in seconds

    #keep track of the items, not very expensive since I only store little numbers of them
    seen = [] #is the array for the detected objects its format is (coords,dimensions,id,and status); status -> 0: not known,1: know, 2:not determined yet
    tmp = dict() #used for remmebeing the matching between the coords and the name and confidence
    confidence_array = [[] for _ in range(100)] #confidence of pictures processed before deciding whether we recognize the pet
    crop_array = [[] for _ in range(100)] #for training purpose



    #cap
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    number = 0
    pet_number = 0 #index for each pet seen during the video
    while cap.isOpened():

        ret, img = cap.read()
        if not ret:
            break
        #width = len(img[0])
        #height = len(img)


        #step 1 first yolo for pet body
        #But we don't have to get rid of the previous rectangle every time we don't detect, we allow a 0.5 sec span of time, and if during this time detect, then update else after
        #this span, get rid of it therefore, to do that we will need to store the time of each detect
        r = detect_s(net_big,meta_big, img, thresh = thresh1, nms = 0.1 )
        r = [(R,P,(x,y,w,h)) for (R,P,(x,y,w,h)) in r if R == b'dog' or R == b'cat']

        #another set of settings
        #change is used for not computing all the time recognition
        change_array = []
        present = [] #used for storing the matching indexes
        yolo_face = [] #used for verbose
        cascade = []
        cascade2 = []
        get_rid = []
        still = []
        #maxi_dist =  (width*0.5 + height *0.5) * 0.30 #30 percent, so for fast moving things, we just do not really recognize well
        var = len(seen)
        #closest points policy (for now)
        # we associate what each current detected center to the previous detected ones, plus we assume that every seen has a match
        
        for i in range(len(seen)):
            idx, dist = closest_pair(seen[i],r,present)
            #if dist >  maxi_dist : #deals with the case when a pet leaves and a pet comes in at the same time, so that we consider the new pet to not be the same as the one that left
            #    print('not reasonable')
            #    continue
            if idx == None: still.append(i)
            present.append((idx,i)) # present stores both the index from array r (where you have the new points) and the index of the matching point to modify in seen array
        
        present1 = [f[0] for f in present] # list of the new indexes in r
        present2 = [f[1] for f in present] # list of the old indexes in previous seen


        cur = time.time()
        for i in range(len(r)):
            if i not in present1: #there is a new pet recognized that appeared
                x,y,w,h = r[i][2]
                x,y,w,h = int(x),int(y),int(w),int(h)
                if [x,y,w,h,cur] not in [s[:5] for s in seen]: 
                    seen.append([x,y,w,h,cur,pet_number,2]) #append to seen x,y,w,h, note when appending we do modify the seen array BUT do not change the order so the matching is still valid
                    pet_number += 1
                tmp[(x,y,w,h)] = (None,math.inf,0)
                change_array.append(True) #as we have detected a new pet, then need to try recognizing him
            else:
                idx = present1.index(i) #this is the index in seen array of the old point to replace
                x,y,w,h = r[i][2]
                x,y,w,h = int(x),int(y),int(w),int(h)
                a,b,c,d,t,pet_nb,recog = seen[present2[idx]]
                if (a,b,c,d) in tmp:
                    name,conf,fcon = tmp[(a,b,c,d)] #make a copy of it before deleting
                    del tmp[(a,b,c,d)] #delete the old key
                    tmp[(x,y,w,h)] = (name,conf,fcon) #   name and conf defined in the previous loop
                seen[present2[idx]] = [x,y,w,h,cur,seen[present2[idx]][5],seen[present2[idx]][6]]
                
                if seen[present2[idx]][6] == 1: change_array.append(False)
                else: change_array.append(True)

        cur = time.time()
        for i in range(var):
            if (i not in present2 or i in still) and cur - seen[i][4] > lapse: get_rid.append(seen[i])
        for X in get_rid: seen.remove(X)
        #print(still)
        #print(pet_number)
        count = 0
        temp_array = [f[:4] for f in seen]
        for (R,P,(x,y,w,h)) in r: #We iterate over what body we see
            x,y,w,h = int(x),int(y),int(w),int(h)
            x1,y1,w1,h1 = x,y,w,h
            if (x,y,w,h) in tmp:
                if tmp[(x,y,w,h)][0] == None: change_array[count] = True 
            
            idx = temp_array.index([x,y,w,h])
            a,b,c,d,t,pet_nb,recog = seen[idx]    
            #print(seen[idx])
            
            #uncentered variables    
            Xc,Yc = x-w//2,y-h//2
            W,H = w,h
            if Xc < 0: Xc = 0
            if Yc < 0: Yc = 0       
            
            # bool to check whether we need to update or not
            if change_array[count] == False:
                #for sanity check, in case I recognized something but turns out to be wrong, we don't want the label to stay unchanged forever
                if random.random() < 0.05: change_array[count] = True

            if change_array[count]:

                next = img[Yc:Yc+h//1,Xc:Xc+w//1]

                #step2 second yolo only for pet face
                r = detect_s(net,meta, next, thresh2)
                #take biggest for now, as we assume the body check gives us separate pets
                if len(r) != 0: (R,P,(x,y,w,h)) = r[0]
                else: continue
                x,y,w,h = int(x),int(y),int(w),int(h)
                X1,Y1 = x-w//2,y-h//2
                if X1 < 0: X1 = 0
                if Y1 < 0: Y1 = 0
                next = next[Y1:Y1+h//1,X1:X1+w//1]
                yolo_face.append((Xc+X1,Yc+Y1,w,h))
                
                #step 4 detect with haarcascade
                gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
                
                width,height  = int((1/2.5)*len(next[0])),int((1/2.5)*len(next)) #We notice some sort of noise with cascade when pixel image too big (haarcascade training set not large dim)
                # we abort the recognition of any pic that has pixel reso of lower than (width,height)
                cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N,minSize = (width,height)) 
                cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF1, minNeighbors=N1,minSize = (width,height))        

                if len(cats_ext) == 0 and len(cats) == 0: 
                    name, neg = model.predict(gray)
                    if recog == 0 and neg > 75: crop_array[pet_nb].append(gray)
                    tmp[(x1,y1,w1,h1)] = name,neg,0.01
                    print('conf1 '+str(neg))
                    number += 1
                else: 
                    if len(cats_ext) != 0:
                        new = sorted(cats_ext,key = lambda A:A[2]*A[3])
                        x,y,w,h = new[-1]
                        x,y = int(x-w*0.1),int(y-h*0.1)
                        if x < 0 : x = 0
                        if y < 0: y = 0
                        w,h = int(1.2*w), int(1.2*h) #trying to extend a bit mroe the rect to see if that helps

                        cascade.append((x+X1+Xc,y+Y1+Yc,w,h))
                        name, neg = model.predict(gray[y:y+h,x:x+w])
                        if recog == 0 and neg >= 60 : crop_array[pet_nb].append(gray[y:y+h,x:x+w])
                        
                        tmp[(x1,y1,w1,h1)] = name,neg,0.3
                        print('conf2 '+str(neg))
                        number += 1
                    elif len(cats) != 0:
                        new = sorted(cats,key = lambda A:A[2]*A[3])
                        x,y,w,h = new[-1]
                        x,y = int(x-w*0.1),int(y-h*0.1)
                        if x < 0 : x = 0
                        if y < 0: y = 0
                        w,h = int(1.2*w), int(1.2*h) #trying to extend a bit mroe the rect to see if that helps

                        cascade2.append((x+X1+Xc,y+Y1+Yc,w,h))
                        

                        name, neg = model.predict(gray[y:y+h,x:x+w])

                        if recog == 0 and neg >= 60 : crop_array[pet_nb].append(gray[y:y+h,x:x+w])
                        
                        #cv2.imwrite('testing/'+str(number)+'file.jpg',gray[y:y+h,x:x+w])
                        tmp[(x1,y1,w1,h1)] = name,neg,0.1
                        print('conf3 '+str(neg))
                        number += 1
                    
                #ALright we have enough crops for this specific pet we can now train on it and add it in our repertoire
                if len(crop_array[pet_nb]) >= 3: 
                    seen[idx][6] = 1
                    full.append('pet'+str(len(full)-1))
                    
                    model = model_training(crop_array[pet_nb],label_numb=len(full)-1,mod=1,load= 'total.xml',name = 'total')
                    with open("PetName.txt",'a') as text: 
                        text.write('\n'+full[-1]) 
                    crop_array[pet_nb] = []
            count += 1

        #print(seen)
        #print(full)
        for index in range(len(seen)):
            obj = seen[index]
            x,y,w,h,t,nb_pet,recog = obj

            
            Xc,Yc = x-w//2,y-h//2
            W,H = w,h
            if Xc < 0: Xc = 0
            if Yc < 0: Yc = 0
            

            if (x,y,w,h) in tmp: 
                name,conf,fcon = tmp[(x,y,w,h)]
                coefs = [element[1] for element in confidence_array[nb_pet]]
                sum_coeff = sum(coefs)
                if sum_coeff == 0: mean = 0
                else: mean = sum([element[0]*element[1] for element in confidence_array[nb_pet]])/sum_coeff
                print(sum_coeff,mean)
                #print(name,conf,fcon)
                if conf < 120 :
                    if conf not in [element[0] for element in confidence_array[nb_pet]]:
                        if sum_coeff < 1: 
                            confidence_array[nb_pet].append([conf,fcon]) #weird values are not taken
                

                if conf < 38 and fcon != 0.01: 
                    #print(name)
                    img = cv2.putText(img,full[name],(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                elif sum_coeff >= 1:

                    if mean <= 50 and (0.3 in coefs or 0.1 in coefs):
                        img = cv2.putText(img,full[name],(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                        seen[index][6] = 1 #known pet
                    else: 
                        seen[index][6] = 0 #unknown pet so we will just give it a label anyways
                        confidence_array[nb_pet] = []
                        
                else: img = cv2.putText(img,'DK',(Xc+10,Yc+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            img = cv2.rectangle(img,(Xc,Yc),(Xc+W,Yc+H),(0,255,0),2)

        if verbose:
            for (x,y,w,h) in yolo_face:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            for (x,y,w,h) in cascade:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            for (x,y,w,h) in cascade2:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,0),2)
        cv2.imshow(window_name,img)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break      








##############################################################################################################################################################





def main(): 
    #here through syntax, you can either train your file/ do videos_yolos/ or vid_yolo
    array = sys.argv[1:]
    if len(array) > 5: print('Too many arguments check -h for usage')
    
    if array[0] == '-m':
        if len(array) != 2: print('Not the right nb of argument, this feature receives only a path/filename')
        else: vid_yolo(array[1])
    elif array[0] == '-r':
        if len(array) != 2 and len(array) != 3: print('Not the right nb of argument, this feature receives only a path/filename')
        if len(array) == 2: video_yolos(array[1])
        if len(array) == 3:
            if array[1] == '-v': video_yolos(array[2],verbose =True)
    elif array[0] == '-l':
        if len(array) != 1: print('Not the right nb of argument, this requires no extra arguments')
        else: print('might implement, not very interesting')
    elif array[0] == '-h':
        if len(array) > 1: print('Not the right nb of argument, this requires no extra arguments')
        else: print('For train use: -t ; for yolo model testing use: -m ; for recognition use: -r')
    else:
        print("argument unknown check -h for usage")
    
        



if __name__ == "__main__":
    live_yolo_recog('lol')


    #vid_yolo("video/vid3.mp4")
    #train_yolos('live_fold',label_numb = 4,name= 'lady')
    #start = time.time()
    #video_yolos('video/vid4.mp4')
    #print(start-time.time())
    
