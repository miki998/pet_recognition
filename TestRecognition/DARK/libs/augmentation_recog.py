from math import *
import tensorflow as tf 
import numpy as np
from keras import *
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import os
import random
import cv2
from matplotlib import pyplot


#This augmentation is reserved for training the recognition net
# We get rid in this augmentation of scaling out, since we don't want to recognize too zoomed in cat
# (our network does it quite well already)

#ATTENTION: We assume that inputs are already cropped face, that's also why we don't zoom in
def aug_recog_files(data,name): #we shift, images is the link to it, filename
    #data is the double array for the image
    height,width = len(data), len(data[0])
    datagen = ImageDataGenerator()
    number = 1
    
    #identity
    gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('train/'+name+'/'+str(number)+name,gray)
    
    # shift left-right
    for _ in range(1):

        shiftx = random.randint(int(-0.2*width),int(0.2*width))
        #print(shiftx)  
        dic = {'ty':shiftx}
        batch = datagen.apply_transform(data,dic)

        # convert to unsigned integers for viewing

        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('train/'+name+'/'+str(number)+name,gray)
        
        number += 1

    # shift up-down
    for _ in range(1):

        shifty = random.randint(int(-0.2*height),int(0.2*height))
        dic = {'tx':shifty}
        
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('train/'+name+'/'+str(number)+name,gray)
        number += 1

    #flip hor   
    for _ in range(1):
        dic = {'flip_horizontal':True}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('train/'+name+'/'+str(number)+name,gray)
        number += 1

    #flip vert
    for _ in range(2):
        dic = {'flip_vertical':True}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('train/'+name+'/'+str(number)+name,gray)
        number += 1
    #rot    
    for _ in range(6):
        rotang = random.randint(-90,90)
        dic = {'theta':rotang}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('train/'+name+'/'+str(number)+name,gray)
        number += 1

    #zooming out so values range from 1 to 2 (not too small cz there is no point )
    for _ in range(4):
        x = random.uniform(1,2)
        y = random.uniform(1,2)
        dic = {'zx':x,'zy':y}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('train/'+name+'/'+str(number)+name,gray)
        number += 1


    #brightness var
    for _ in range(4):
        bright = random.random()
        dic = {'brightness':bright}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('train/'+name+'/'+str(number)+name,gray)
        number += 1


##############################################################################################################################################################################




def aug_recog(data): #----> output array of augmented crops
    #data is the double array for the image
    height,width = len(data), len(data[0])
    datagen = ImageDataGenerator()
    number = 1
    crop_array = []
    #identity
    gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
    crop_array.append(gray)
    
    
    # shift left-right
    for _ in range(4):
        
        shiftx = random.randint(int(-0.2*width),int(0.2*width))
        #print(shiftx)
        dic = {'ty':shiftx}
        batch = datagen.apply_transform(data,dic)
        
        # convert to unsigned integers for viewing
        gray = cv2.cvtColor(batch,cv2.COLOR_BGR2GRAY)
        crop_array.append(gray)
        
        number += 1
    
    # shift up-down
    for _ in range(4):
        
        shifty = random.randint(int(-0.2*height),int(0.2*height))
        dic = {'tx':shifty}
        
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(batch,cv2.COLOR_BGR2GRAY)
        crop_array.append(gray)
        number += 1

    #flip hor
    for _ in range(1):
        dic = {'flip_horizontal':True}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(batch,cv2.COLOR_BGR2GRAY)
        crop_array.append(gray)
        number += 1
    
    #flip vert
    for _ in range(1):
        dic = {'flip_vertical':True}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(batch,cv2.COLOR_BGR2GRAY)
        crop_array.append(gray)
        number += 1
    #rot
    for _ in range(4):
        rotang = random.randint(-90,90)
        dic = {'theta':rotang}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(batch,cv2.COLOR_BGR2GRAY)
        crop_array.append(gray)
        number += 1
    
    #zooming out so values range from 1 to 2 (not too small cz there is no point )
    for _ in range(4):
        x = random.uniform(1,2)
        y = random.uniform(1,2)
        dic = {'zx':x,'zy':y}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(batch,cv2.COLOR_BGR2GRAY)
        crop_array.append(gray)
        number += 1
    
    
    #brightness var
    for _ in range(2):
        bright = random.random()
        dic = {'brightness':bright}
        batch = datagen.apply_transform(data,dic)
        gray = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        crop_array.append(gray)
        number += 1

    return crop_array



##############################################################################################################################################################################



if __name__ == "__main__":
    
    for f in os.listdir('video/'):
        if f.split('.')[-1] == 'jpg':
            img = cv2.imread('video/'+f)
            aug_recog_files(img,f)
    
