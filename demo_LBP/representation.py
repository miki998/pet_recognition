
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import time
from keras.preprocessing.image import ImageDataGenerator
from math import *


#####################################################################################################################

""" useful functions """ 

# settings for LBP
radius = 1
n_points = 8 * radius
METHOD = 'uniform'


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=False, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')




####################################################################################################

""" Main function for representing the lbp images """


def main(filepath):
    img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    if len(img[0])>100 and len(img) > 100: img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
    else: img = cv2.resize(img,(100,100),interpolation=cv2.INTER_CUBIC)

    new = local_binary_pattern(img,n_points,radius,method=METHOD)
    lbp = 256*new/25 #just to make the illustration more beautiful, colors are totally arbitrary

    width, height = len(img[0]),len(img)
    div = 3
    unitw, unith = width//3+1, height//3+1

    fig = plt.figure()
    ax1 = fig.add_subplot(4,4,1)
    ax1.imshow(img,cmap='gray')
    ax2 = fig.add_subplot(4,4,2)
    ax2.imshow(lbp,cmap='gray')

    #resized = cv2.resize(img,(400,400),interpolation=cv2.INTER_AREA)
    #new1 = local_binary_pattern(resized,n_points,radius,method=METHOD)

    #ax3 = fig.add_subplot(4,4,15)
    #ax3.imshow(256*new1/25,cmap='gray')
    arrays = []
    for i in range(div):
        for j in range(div):
            arrays.append(new[i:(i+1)*unitw,j:(j+1)*unith])

    for i in range(div**2):
        ax = fig.add_subplot(4,4,3+i)
        
        hist(ax,arrays[i])
    plt.show()




###############################################################################################################


if __name__ == '__main__':
    array = sys.argv[1:]
    if len(array) !=1 :
        print('Not the right argument number')
        sys.exit(0)
    else:
        if not os.path.exists(array[0]): 
            print('path does not exist')
            sys.exit(0)
        else:
            main(array[0])
