import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import cv2
import sys
import re
import pickle
import time
from os import listdir
from copy import deepcopy
from math import *

sys.path.insert(0, 'DARK/python/')
from darknet import *

###########################################################################################################

""" All the nice classes here """

class Yoloface:

	def __init__(self,thresh=0.2,nms=0.1):
		self.net = load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "DARK/weights/front_prof2_200k.weights".encode("utf-8"), 0)
		self.meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
		self.thresh = thresh
		self.nms = nms
			
	def detect(self,img):
		R = detect_s(self.net,self.meta,img,thresh=self.thresh,nms=self.nms)
		if len(R) == 0: 
			print('no damn thing detected')
			sys.exit(0)
		return R

class Yolobody:

	def __init__(self,thresh=0.2,nms=0.1):
		self.net = load_net("DARK/cfg/cat-dog-yolov3-tiny.cfg".encode("utf-8"), "DARK/weights/body_500k.weights".encode("utf-8"), 0)
		self.meta = load_meta("DARK/cfg/cat-dog-obj.data".encode('utf-8'))
		self.thresh = thresh
		self.nms = nms
			
	def detect(self,img):
		R = detect_s(self.net,self.meta,img,thresh=self.thresh,nms=self.nms)
		if len(R) == 0: 
			print('no damn thing detected')
			sys.exit(0)
		return R

class Yoloeye:

	def __init__(self,thresh=0.2,nms=0.1):
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


class HaarCascadeExt:
	def __init__(self,SF=1.03,N=6):
		self.cat_ext_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
		self.SF = SF
		self.N = N
	
	def detect(self,img,dim):
		R = self.cat_ext_cascade.detectMultiScale(img,scaleFactor=self.SF,minNeighbors=self.N,minSize=dim)
		return R


class Model:
	def __init__(self,filename=None,mod=1):
		
		if mod == 1:self.model = cv2.face.LBPHFaceRecognizer_create()
		elif mod == 2 :self.model = cv2.face.FisherFaceRecognizer_create()
		elif mod == 3 :self.model = cv2.face.EigenFaceRecognizer_create()
		if filename:
			if os.path.exists('models/'+filename):  self.model.read('models/'+filename)
			else: print("Alright, you didn't input a correct model so we will start from scratch")
		else: print("Alright, you didn't input a model so we will start from scratch")
		self.filename = filename
		self.mod = mod    

	def prediction(self,img):

		if len(img.shape) == 3: return self.model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
		return self.model.predict(img)
		
	def training(self,array,label_nb): #Might need to do some resizing here
		Training_Data, Labels = [] , [] 
		for im in array:
			# in case turn into grayscale
			if len(im.shape)==3: im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			Training_Data.append( np.asarray( im, dtype=np.uint8))
			Labels.append(label_nb)
		print('Gathering Data to train on !')
		Labels = np.asarray(Labels, dtype=np.int32) #I mean, i guess database won't exceed 256 for now
			
		face = np.asarray( Training_Data)
		if self.mod == 1:
			if self.filename: self.model.update(face,Labels) 
			else: self.model.train(face,Labels)
		else: 
			try: self.model.train(face,Labels) 
			except: sys.exit(0) 
		print("Model updated sucessefully, Congratulations")
		if self.filename:
			try :self.model.save('models/'+self.filename)
			except: print('Could not save the model')
		else:
			try :self.model.save('models/standard.xml')
			except: print('Could not save the model')

def cent2rect(x,y,w,h):
	x,y,w,h = int(x),int(y),int(w),int(h)
	x,y = x-w//2,y-h//2
	if x < 0: x = 0
	if y < 0: y = 0
	return x,y,w,h


def add_rect(img,bbox):
	x,y,w,h = bbox
	tmp = deepcopy(img)	
	im = cv2.rectangle(tmp, (x,y), (x+w,y+h), (255, 0, 0), 2) 
	return im
