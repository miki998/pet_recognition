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
from copy import deepcopy
from tqdm import tqdm


""" convert and deconvert for annotation  Yolo """
def convert(size, box):

    x = box[0]/1.0 + (box[2])/2.0
    y = box[1]/1.0 + box[3]/2.0
    x = x/size[0]
    y = y/size[1]

    w = box[2]/size[0]
    h = box[3]/size[1]

    return (x,y,w,h)

def deconvert(img_w,img_h,annbox):
    #x,y is top left corners 
    centx, centy = int(annbox[0] * img_w), int(annbox[1] * img_h)
    w,h = annbox[2] *img_w, annbox[3] * img_h
    x,y = centx-w/2, centy-h/2
    return [int(x),int(y),int(w),int(h)]




################################################################################################################################################################



"""this is supposed to be a script that helps augmenting the data amount, it is not used by the customer (so does not catch error/exception)

The first function shows how everything else works, I decided to repeat myself so and thus separate the functions for augmentations as we do not and should not use everything 
for any case. ex: don't want to have a zooming in when we want a standard size to capture, or no brightness anyways since gray so not very efficient and might just give identities 
and thus higher weight for useless cases.
"""

def shift_image_LR(file,cur,verbose = False): #we shift, images is the link to it, filename
    img = load_img(file)
    filename = file.split('/')[-1]
    data = img_to_array(img)
    height,width = len(data), len(data[0])
    datagen = ImageDataGenerator()
    number = cur + 1

    #Opening files 
    textfile = filename.split('.')[0]
    f = open('Images/full_catAndDog/'+textfile+'.txt','r')
    contents = f.readlines()
    test_list = open('DARK/order/cat-dog-test.txt','a')    
    train_list = open('DARK/order/cat-dog-train.txt','a')

    #create new txt file that gives new coordinates to the augmentation that I made
    m = open('testing/new'+str(number)+textfile+'.txt','w')


    #actually saving the the modified picture
    shiftx = random.randint(int(-0.2*width),int(0.2*width))
    dic = {'ty':shiftx}
    batch = datagen.apply_transform(data,dic)
    img = deepcopy(batch)
    batch = batch.astype('uint8')
    pyplot.imsave('testing/new'+str(number)+filename,batch)


    #now we deal with the content of the file we created
    for i in range(len(contents)):
        if contents[i].strip() == '': continue
        nclass,a,b,c,d = contents[i].split()
        nclass, a, b,c,d = int(nclass), float(a),float(b),float(c),float(d)
        a,b,c,d = deconvert(width,height,[a,b,c,d]) #convert and deconvert functions are defined on top, they are format fitting functions

        newa = a - shiftx #the shifting for the bounding box

        if verbose: img = cv2.rectangle(img,(newa,b),(newa+c,b+d),(255,0,0),2) #Draw the rectangles

        newa,newb,newc,newd = convert((width,height),[newa,b,c,d]) #convert back to the format for txt files
        m.write(str(nclass)+' '+ str(newa)+ ' '+str(newb)+' '+str(newc)+' '+str(newd)+'\n')

    #randomly assign the picture to either train or test.txt
    if random.random() < 0.05: test_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    else: train_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    
    if verbose: cv2.imwrite('testing/vnew'+str(number)+textfile+'.jpeg',img) #writing out the pictures with multiple rectangles
    
    #closing up folders
    train_list.close()
    test_list.close()
    f.close()
    m.close()
    return number
    


################################################################################################################################################################
    



def shift_image_UD(file,cur,verbose = False): #we shift, images is the link to it, filename
    img = load_img(file)
    filename = file.split('/')[-1]
    data = img_to_array(img)
    height,width = len(data), len(data[0])
    datagen = ImageDataGenerator()
    number = cur + 1

    #Opening files 
    textfile = filename.split('.')[0]
    f = open('Images/full_catAndDog/'+textfile+'.txt','r')
    contents = f.readlines()
    test_list = open('DARK/order/cat-dog-test.txt','a')    
    train_list = open('DARK/order/cat-dog-train.txt','a')

    #create new txt file that gives new coordinates to the augmentation that I made
    m = open('testing/new'+str(number)+textfile+'.txt','w')


    #actually saving the the modified picture
    shiftx = random.randint(int(-0.2*height),int(0.2*height))
    dic = {'tx':shiftx}
    batch = datagen.apply_transform(data,dic)
    img = deepcopy(batch)
    batch = batch.astype('uint8')
    pyplot.imsave('testing/new'+str(number)+filename,batch)


    #now we deal with the content of the file we created
    for i in range(len(contents)):
        if contents[i].strip() == '': continue
        nclass,a,b,c,d = contents[i].split()
        nclass, a, b,c,d = int(nclass), float(a),float(b),float(c),float(d)
        a,b,c,d = deconvert(width,height,[a,b,c,d])

        newb = b - shiftx #the shifting for the bounding box

        if verbose: img = cv2.rectangle(img,(a,newb),(a+c,newb+d),(255,0,0),2) #Draw the rectangles

        newa,newb,newc,newd = convert((width,height),[a,newb,c,d]) #convert back to the format for txt files
        m.write(str(nclass)+' '+ str(newa)+ ' '+str(newb)+' '+str(newc)+' '+str(newd)+'\n')

    #randomly assign the picture to either train or test.txt
    if random.random() < 0.05: test_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    else: train_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    
    if verbose: cv2.imwrite('testing/vnew'+str(number)+textfile+'.jpeg',img) #writing out the pictures with multiple rectangles   
    
    train_list.close()
    test_list.close()
    m.close()
    return number




################################################################################################################################################################





def identity(file,cur,verbose = False):
    img = load_img(file)
    filename = file.split('/')[-1]
    data = img_to_array(img)
    height,width = len(data), len(data[0])
    number = cur + 1
    textfile = filename.split('.')[0]
    f = open('Images/full_catAndDog/'+textfile+'.txt','r')
    contents = f.readlines()
    test_list = open('DARK/order/cat-dog-test.txt','a')    
    train_list = open('DARK/order/cat-dog-train.txt','a')

    img = deepcopy(data)

    data = data.astype('uint8')
    pyplot.imsave('testing/new'+str(number)+filename,data)
    m = open('testing/new'+str(number)+textfile+'.txt','w')

    for i in range(len(contents)):
        if contents[i].strip() == '': continue
        nclass,a,b,c,d = contents[i].split()
        a,b,c,d = float(a),float(b),float(c),float(d)
        a,b,c,d = deconvert(width,height,[a,b,c,d])
        
        if verbose: img = cv2.rectangle(img,(a,b),(a+c,b+d),(255,0,0),2)


        newa,newb,newc,newd = convert((width,height),[a,b,c,d])
        m.write(str(nclass)+' '+ str(newa)+ ' '+str(newb)+' '+str(newc)+' '+str(newd)+'\n')
    
    #randomly assign the picture to either train or test.txt
    if random.random() < 0.05: test_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    else: train_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')    

    if verbose: cv2.imwrite('testing/vnew'+str(number)+textfile+'.jpeg',img)

    test_list.close()
    train_list.close()
    m.close()
    return number




################################################################################################################################################################




def flip_image_hor(file,cur,verbose = False):

    img = load_img(file)
    filename = file.split('/')[-1]
    data = img_to_array(img)
    height,width = len(data), len(data[0])
    datagen = ImageDataGenerator()
    number = cur + 1
    
    textfile = filename.split('.')[0]
    f = open('Images/full_catAndDog/'+textfile+'.txt','r')
    contents = f.readlines()
    test_list = open('DARK/order/cat-dog-test.txt','a')    
    train_list = open('DARK/order/cat-dog-train.txt','a')


    dic = {'flip_horizontal':True}
    batch = datagen.apply_transform(data,dic)
    m = open('testing/new'+str(number)+textfile+'.txt','w')
    img = deepcopy(batch)
    batch = batch.astype('uint8')
    pyplot.imsave('testing/new'+str(number)+filename,batch)
    
    for i in range(len(contents)):
        if contents[i].strip() == '': continue
        nclass,a,b,c,d = contents[i].split()
        nclass, a, b,c,d = int(nclass), float(a),float(b),float(c),float(d)
        a,b,c,d = deconvert(width,height,[a,b,c,d])

        newa = width - a - c
        if verbose: img = cv2.rectangle(img,(newa,b),(newa+c,b+d),(255,0,0),2)

        newa,newb,newc,newd = convert((width,height),[newa,b,c,d])
        m.write(str(nclass)+' '+ str(newa)+ ' '+str(newb)+' '+str(newc)+' '+str(newd)+'\n')

    #randomly assign the picture to either train or test.txt
    if random.random() < 0.05: test_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    else: train_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg') 

    if verbose: cv2.imwrite('testing/vnew'+str(number)+textfile+'.jpeg',img)
    
    m.close()
    test_list.close()
    train_list.close()
    return number




################################################################################################################################################################




def flip_image_ver(file,cur,verbose = False):

    img = load_img(file)
    filename = file.split('/')[-1]
    data = img_to_array(img)
    height,width = len(data), len(data[0])
    datagen = ImageDataGenerator()
    number = cur + 1
    

    textfile = filename.split('.')[0]
    f = open('Images/full_catAndDog/'+textfile+'.txt','r')
    contents = f.readlines()
    test_list = open('DARK/order/cat-dog-test.txt','a')    
    train_list = open('DARK/order/cat-dog-train.txt','a')

    dic = {'flip_vertical':True}
    batch = datagen.apply_transform(data,dic)
    img = deepcopy(batch)
    batch = batch.astype('uint8')
    pyplot.imsave('testing/new'+str(number)+filename,batch)
            
    m = open('testing/new'+str(number)+textfile+'.txt','w')

        
    for i in range(len(contents)):
        if contents[i].strip() == '': continue
        nclass,a,b,c,d = contents[i].split()
        nclass, a, b,c,d = int(nclass), float(a),float(b),float(c),float(d)
        a,b,c,d = deconvert(width,height,[a,b,c,d])
        newb = height - b - d

        if verbose: img = cv2.rectangle(img,(a,newb),(a+c,newb+d),(255,0,0),2)

        newa,newb,newc,newd = convert((width,height),[a,newb,c,d])


        m.write(str(nclass)+' '+ str(newa)+ ' '+str(newb)+' '+str(newc)+' '+str(newd)+'\n')


    if verbose: cv2.imwrite('testing/vnew'+str(number)+textfile+'.jpeg',img)
    
    #randomly assign the picture to either train or test.txt
    if random.random() < 0.05: test_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    else: train_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')

    train_list.close()
    test_list.close()
    m.close()
    return number 




################################################################################################################################################################


"""Turn the center of the image to (0,0) """

def normalize(pts,height,width):
    x,y = pts
    x -= width/2
    y = height/2 - y
    return x,y

def back(pts,height,width):
    x,y = pts
    x += width/2
    y = height/2 - y
    return x,y



################################################################################################################################################################ 




def rotation_image(file,cur,verbose = False):

    img = load_img(file)
    filename = file.split('/')[-1]
    data = img_to_array(img)
    height,width = len(data), len(data[0])
    datagen = ImageDataGenerator()

    number = cur + 1
    
    textfile = filename.split('.')[0]
    f = open('Images/full_catAndDog/'+textfile+'.txt','r')
    contents = f.readlines()
    test_list = open('DARK/order/cat-dog-test.txt','a')    
    train_list = open('DARK/order/cat-dog-train.txt','a')
    m = open('testing/new'+str(number)+textfile+'.txt','w')

    rotang = random.randint(-90,90)
    dic = {'theta':rotang}
    batch = datagen.apply_transform(data,dic)
    img = deepcopy(batch)
    # convert to unsigned integers for viewing
    batch = batch.astype('uint8')
    pyplot.imsave('testing/new'+str(number)+filename,batch)

    rotang = -rotang #the rotation angle is the unconventional one for the apply transform... which is very stupid

    for i in range(len(contents)):
        if contents[i].strip() == '': continue
        nclass,a,b,c,d = contents[i].split()
        nclass, a, b,c,d = int(nclass), float(a),float(b),float(c),float(d)
        a,b,c,d = deconvert(width,height,[a,b,c,d])
        

        fr_points = [(a,b),(a+c,b),(a+c,b+d),(a,b+d)]
        results = []
            
        for pts in fr_points:
            X,Y = pts
            x,y = normalize((X,Y),height,width)


            u = np.array([x,y])
            rot_mat = np.matrix([[cos(radians(rotang)),-sin(radians(rotang))],[sin(radians(rotang)),cos(radians(rotang))]])

            R = np.dot(rot_mat,u)
            w,z = R[0,0],R[0,1]
                
            e,f = back((w,z),height,width)
            results.append((e,f))

        #find the min and max to create a non-tilted rectangle
        newa,newb = min(results, key = lambda x: x[0]),min(results, key = lambda x: x[1])
        newa,newb = newa[0],newb[1]
        newc = (max(results, key = lambda x: x[0])[0]-newa)
        newd = ( max(results, key = lambda x: x[1])[1]-newb)

        if verbose: img = cv2.rectangle(img,(int(newa),int(newb)),(int(newa+newc),int(newb+newd)),(255,0,0),2)

        #then write
        newa,newb,newc,newd = convert((width,height),[newa,newb,newc,newd])
        m.write(str(nclass)+' '+ str(newa)+ ' '+str(newb)+' '+str(newc)+' '+str(newd)+'\n')

    #randomly assign the picture to either train or test.txt
    if random.random() < 0.05: test_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    else: train_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    
    if verbose: cv2.imwrite('testing/vnew'+str(number)+textfile+'.jpeg',img)

    m.close()
    train_list.close()
    test_list.close()
    return number



################################################################################################################################################################




def scale_image(file,cur,verbose = False):
    
    img = load_img(file)
    filename = file.split('/')[-1]
    data = img_to_array(img)
    height,width = len(data), len(data[0])
    datagen = ImageDataGenerator()

    number = cur + 1

    textfile = filename.split('.')[0]
    f = open('Images/full_catAndDog/'+textfile+'.txt','r')
    contents = f.readlines()
    test_list = open('DARK/order/cat-dog-test.txt','a')    
    train_list = open('DARK/order/cat-dog-train.txt','a')
    m = open('testing/new'+str(number)+textfile+'.txt','w')

    X = random.uniform(0.5,2)
    Y = X

    dic = {'zx':Y,'zy':X}
    batch = datagen.apply_transform(data,dic)
    # convert to unsigned integers for viewing
    img = deepcopy(batch)
    batch = batch.astype('uint8')
    pyplot.imsave('testing/new'+str(number)+filename,batch)


    for i in range(len(contents)):
        if contents[i].strip() == '': continue
        nclass,a,b,c,d = contents[i].split()
        nclass, a, b,c,d = int(nclass), float(a),float(b),float(c),float(d)
        a,b,c,d = deconvert(width,height,[a,b,c,d])

        x,y = normalize((a,b),height,width)
        x1,y1 = normalize((a,b+d),height,width)
        x2,y2 = normalize((a+c,b),height,width)

        x,y = int(x/X),int(y/Y)
        x1,y1, = int(x1/X),int(y1/Y)
        x2,y2, = int(x2/X),int(y2/Y)
        w,z = x2 - x, abs(y1 - y) 

        x,y = back((x,y),height,width)
        x,y = int(x),int(y)
        if verbose: img = cv2.rectangle(img,(x,y),(x+w,y+z),(255,0,0),2)
        x,y,w,z = convert((width,height),[x,y,w,z])
            
        m.write(str(nclass)+' '+ str(x)+ ' '+str(y)+' '+str(w)+' '+str(z)+'\n')

    if verbose: cv2.imwrite('testing/vnew'+str(number)+textfile+'.jpeg',img)

    #randomly assign the picture to either train or test.txt
    if random.random() < 0.05: test_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    else: train_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')

    m.close()
    f.close()
    train_list.close()
    test_list.close()
    return number




################################################################################################################################################################




def bright_image(file,cur,verbose = False):
    img = load_img(file)
    filename = file.split('/')[-1]
    data = img_to_array(img)    
    height,width = len(data), len(data[0])
    datagen = ImageDataGenerator()


    number = cur + 1
    textfile = filename.split('.')[0]
    f = open('Images/full_catAndDog/'+textfile+'.txt','r')
    contents = f.readlines()
    m = open('testing/new'+str(number)+textfile+'.txt','w')
    test_list = open('DARK/order/cat-dog-test.txt','a')    
    train_list = open('DARK/order/cat-dog-train.txt','a')

    dic = {'brightness':random.random()}
    batch = datagen.apply_transform(data,dic)
    img = deepcopy(batch)
    # convert to unsigned integers for viewing
    batch = batch.astype('uint8')
    pyplot.imsave('testing/new'+str(number)+filename,batch)


    for i in range(len(contents)):
        if contents[i].strip() == '': continue
        nclass,a,b,c,d = contents[i].split()
        nclass, a, b,c,d = int(nclass), float(a),float(b),float(c),float(d)
        a,b,c,d = deconvert(width,height,[a,b,c,d])
        
        if verbose: img = cv2.rectangle(img,(int(a),int(b)),(int(a+c),int(b+d)),(255,0,0),2)

        #write
        newa,newb,newc,newd = convert((width,height),[a,b,c,d])
            
        m.write(str(nclass)+' '+ str(newa)+ ' '+str(newb)+' '+str(newc)+' '+str(newd)+'\n')


    #randomly assign the picture to either train or test.txt
    if random.random() < 0.05: test_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')
    else: train_list.write('\nImages/full_catAndDog/new'+str(number)+textfile+'.jpg')

    if verbose: cv2.imwrite('testing/vnew'+str(number)+textfile+'.jpeg',img)
    m.close()
    f.close()
    train_list.close()
    test_list.close()
    return number



################################################################################################################################################################




if __name__ == "__main__":
    number = 0 
    
    for f in tqdm(os.listdir('Images/full_catAndDog/')):
        if f.split('.')[-1] != 'txt':
            #number = identity('Images/full_catAndDog/'+f,number,True)
            for _ in range(1):
                number = shift_image_LR('Images/full_catAndDog/'+f,number,True)
                number = shift_image_UD('Images/full_catAndDog/'+f,number,True)
                
            number = flip_image_ver('Images/full_catAndDog/'+f,number,True)
            number = flip_image_hor('Images/full_catAndDog/'+f,number,True)
            #for _ in range(6): number = rotation_image('Images/full_catAndDog/'+f,number,True) #Note that the rotation, the cropping box is pretty much an upper bounding due to... hehe
            
            for _ in range(1): number = bright_image('Images/full_catAndDog/'+f,number,True)
            #for _ in range(3): number = scale_image('Images/full_catAndDog/'+f,number,True)

            #scaling is STUPID DK WHO WROTE THIS STUPID LIBRARY Not needed in our current training session for our model, but bodywise will need it 
            #same for rotation, super unintuitive, test carefully before applying to train, it is useless, pointless and stupidly bad written
    
