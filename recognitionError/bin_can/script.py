import os
import random

array = [f for f in os.listdir('tmp/body')]
array1 = [f for f in os.listdir('tmp/face')]
array2 = [f for f in os.listdir('tmp/haar')]

for _ in range(100):
    img,img1,img2 = random.choice(array),random.choice(array1),random.choice(array2)
    array.remove(img);array1.remove(img1);array2.remove(img2)
    os.rename('tmp/body/'+img,'sample/body/'+img)
    os.rename('tmp/face/'+img1,'sample/face/'+img1)
    os.rename('tmp/haar/'+img2,'sample/haar/'+img2)

