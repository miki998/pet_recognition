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
from keras.models import load_model


IMAGE_SIZE = 512
#Please do not change the name of the file
""" Auxiliary part, never use alone """
def auxiliary(cap,window_name,frame = None):
    ret, mid = cap.read()
    if frame == None:
        try:
            cv2.imshow(window_name,mid)
        except:
            print('Could not show the file')
    else:
        cats_ext,array,n,name,conf,verf = frame
        
        cur = 0
        n = len(array)
        for (x,y,w,h) in cats_ext:
            #if w < minWidth or h < minHeight : continue
            X,Y = x-10, y-10
            if cur >= n: break
            name,conf,verf = array[cur]
            if verf:
                img = cv2.putText(mid,str(name)+'conf:'+str(conf),(X,Y), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            else:
                img = cv2.putText(mid,'DK',(X,Y), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            img = cv2.rectangle(mid,(x,y),(x+w,y+h),(0,255,0),2)
            cur += 1
    c = cv2.waitKey(10)
    if c & 0xFF == ord('q'):
        return

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x: int(re.findall(r'\d+', x)[-1]))
     
    print('Getting frames composing video')
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)

        #inserting the frames into an image array
        frame_array.append(img)
 
    try:
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    except:
        print('no video written')
    for i in range(len(frame_array)):

        # writing to a image array
        out.write(frame_array[i])
    out.release()

def generate(img,cat_cascade,cat_ext_cascade,SF=1.03,N=3, live = None, model = None): #Input: image (double array format) Output: gray cropped images
    #This generate is solely used for live recognition and replace process's work to a non writing mode
    cropped_face = []

    if live != None: 
        cap, window_name, frame = live
        auxiliary(cap,window_name,frame)

    # convery to gray scale since input from video is colored
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #To rule out small rectangles
    height, width, channels = img.shape
    minHeight, minWidth = height//12, width//12

    # this function returns tuple rectangle starting coordinates x,y, width, height
    #cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N) 
    cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N,minSize = (minHeight,minWidth))

    if live != None: auxiliary(cap,window_name,frame)
    
    print('Sorting out which gray Cropped face to take, and resizing them')
    for (x,y,w,h) in cats_ext:
        if model != None:
            check = cv2.resize(img[y:y+h,x:x+w],(IMAGE_SIZE,IMAGE_SIZE),interpolation = cv2.INTER_AREA)
            checks = model.predict(np.array([check]))
            print(checks)
            if checks[0][0] < 0.95: 
                print(0)
                continue
        if w < minWidth or h < minHeight : continue
        #cropped faces
        gray = cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(400,400)) #400,400 is a standard size
        cropped_face.append(gray)
 
    
    return cropped_face #So this is an array of images

"""From here you can use separately"""

def processImage(image_dir,image_filename,pathOut="test_images/cats/",crop = False, one_crop = False): #path from petDetec folder
    cropped_face = []
    dir_path = os.path.dirname(os.path.abspath('cat.py')) + '/'
    #tunable parameters 
    SF = 1.03 #scale factors
    N = 3 #minimum neighors
    
    #cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface.xml')
    cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')


    # read the image
    img = cv2.imread(image_dir+'/'+image_filename)
    # convery to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #To rule out small rectangles
    height, width, channels = img.shape
    minHeight, minWidth = height//12, width//12

    # this function returns tuple rectangle starting coordinates x,y, width, height
    #cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize = (minHeight,minWidth))
    #print(cats) 
    cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize = (minHeight,minWidth))
    #print(cats_ext)
    
    
    # draw a blue rectangle on the image
    for (x,y,w,h) in cats_ext:
        #Use classifie to check if cat or not  and draw or not rectangle
        #check = cv2.resize(img[y:y+h,x:x+w],(IMAGE_SIZE,IMAGE_SIZE),interpolation = cv2.INTER_AREA)
        #checks = model.predict(np.array([check]))
        #print(checks)
        #if checks[0][0] < 0.99:
        #    print(0)
        #    continue
        if w < minWidth or h < minHeight : continue
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #cropped face if wants
        if crop:
            gray = cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,(400,400)) #400,400 is a standard size
            cropped_face.append(gray)

    # draw a green rectangle on the image 
    #for (x,y,w,h) in cats_ext:
    #    if w < minWidth or h < minHeight : continue
    #    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    if pathOut != None:
        # save the image to a file
        try:
            cv2.imwrite(dir_path+pathOut+'out'+image_filename,img)
        except:
            print('No file written')
    
    if one_crop:
        #Take the biggest rectangle
        try: 
            return [max(cropped_face, key=len)]
        except:
            print('This frame has nothing detected')
    return cropped_face #So this is an array of images


def live_detection(window_name, camera_idx = 0, model = None ):
    #tunable parameters 
    SF = 1.03 #scale factors
    N = 3 #minimum neighors
    #cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface.xml')
    cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')


    #setting the cam window
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(camera_idx)
    
    print('Camera ready to use')
    while cap.isOpened():
        ok, img = cap.read() # capture a frame
        
    
        
        if not ok:           
            break         
        height, width, channels = img.shape
        minHeight, minWidth = height//12, width//12
        # transform the frame into grey image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        
        cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N, minSize = (minHeight,minWidth))
        
        for (x,y,w,h) in cats_ext:
            if model != None:
                check = cv2.resize(img[y:y+h,x:x+w],(IMAGE_SIZE,IMAGE_SIZE),interpolation = cv2.INTER_AREA)
                checks = model.predict(np.array([check]))
                print(checks)
                if checks[0][0] < 0.99: 
                    print(0)
                    continue
            if w < minWidth or h < minHeight : continue
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.imshow(window_name,img)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break       
    # When everything done, release the capture
    print('Done using')
    cap.release()
    cv2.destroyAllWindows()

# we try here to do similarly to live detec, and just recreate the video but with detection mark drawn
# split frames -> draw -> recombine frames 
def video_detection(filename, fps = 25.0,pathIn = 'aux_out/data', pathOut='aux_out/changeData'): #with extension 
    #model = load_model('classifier_model/eightmodel.h5')
    try:
        cap = cv2.VideoCapture(filename)
    except:
        print('target file is not on the same directory')

    try:
        if not os.path.exists(pathIn):
            os.makedirs(pathIn)
                
    except OSError:
        print ('Error: Creating directory of data')
    #Before sending to pahtIn folder, we will delete everything in the pathIn folder
    filelist = [f for f in os.listdir(pathIn)]
    for f in filelist: os.remove(pathIn+'/'+f) 
    print('Getting all the frames from video')
    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
        
        # Firstly splitting
        name = pathIn+'/frame' + str(currentFrame) + '.jpg'
        cv2.imwrite(name, frame)
        currentFrame += 1   # To stop duplicate images
    
    if not os.path.exists(pathOut): os.makedirs(pathOut)

    #Before sending to pahtIn folder, we will delete everything in the pathIn folder
    filelist = [f for f in os.listdir(pathOut)]
    for f in filelist: os.remove(pathOut+'/'+f) 

    print('Processing the images')
    #Drawing rectangles
    filelist = [f for f in os.listdir(pathIn)]
    for f in filelist: processImage(pathIn+'/',f,pathOut+'/')
        
    #video directory must always be in the same as this file
    try:
        if not os.path.exists('video'):
            os.makedirs('video')
    except OSError:
        print('Error: Creating directory of video')

    print('We are almost done, recombining the images')
    #now recombining them to get a video
    #saving the video under the name:= "new" + filename
    filename = filename.split('/')[-1]
    outname = "video/"+"new"+filename
    convert_frames_to_video(pathOut+'/',outname,fps)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_detection("video/vid5high.mp4")
    
