import matplotlib.pyplot as plt
import cv2

img = cv2.imread('../../result/predictions.jpg')
plt.imshow(img[:,:,::-1])
plt.show()
