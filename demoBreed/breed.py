import numpy as np
from keras.applications import resnet50
from keras.preprocessing import image
import os
import time

model = resnet50.ResNet50()

def run_model(img_path):
    img = image.load_img(path=img_path, target_size=(224,224))
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    X = resnet50.preprocess_input(X)
    X_Pred = model.predict(X)
    return resnet50.decode_predictions(X_Pred, top = 1)

if __name__ == '__main__':
    for file in os.listdir('images'):
        for _,name, likelihood in run_model('images/'+file)[0]:
            print("predictions for: "+file+' '+ name+ 'with: '+str(likelihood))
        print()
        time.sleep(3)