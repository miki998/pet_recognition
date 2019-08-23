# Cat-Dog-face-recognition

The goal being to be able to stably detect cat or dog, then either dynamically or statically recognize the pet that we have detected in a pre-given database or a database that we update 
by labelling as live filming goes. 

## General Prerequisites

    python3
    jupyter notebook installed (either through conda or downloaded)
    keras 
    openCV (>= 4.1.0)
    tensorflow (gpu version is optional) 
    CUDA (used: 10.1 any version that is compatible with your GPU (Optional))
    tkinter 
    haarcascade weights (if you have downloaded openCV in weird way)
    matplotlib
    PIL (Image)


## Structure:

Here we give a general explanation of the structure, more detail can be found on each folder (READMEs)
- backupScript: just some previous version folder 
- backupWeight: has all the different weights/models needed for the project
- CNNCompare: CNN method to use as comparison
- demoBreed: demo to see the efficiency of a lib model for breed recognition
- demo_detect: demo to test out the weights from yolo detection
- demo_eye_track: demo to see yolo detection of eyes and result of horizontalizing according to that detection
- demo_LBP: demo to display lbp features of pictures
- Method_CascadaYolo: main folder, do recognition and detection with a starting/fix model 
- OwnModel: implementation of lpbh algorithm using own idea
- Simult_recog_train: main folder, do recognition and detection. Models can be overwritten and updated as process goes
- TestRecognition: LBPH algo from opencv test, the recognition of an image starting from a pre-trained model (training available too inside the folder)
- tool_checking_corrupt: for corrupted pics, dump bad pictures
- YoloTraining: script for a simple training (fixed architecture) of yolo model (detection purpose)
- YoloV3-Annotation-Tool: Annotation Tool to bbox label images

Ideas will be detailed inside each single folders.


## SOURCES/ CREDITS

    https://github.com/pjreddie/darknet 
    https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
    https://github.com/ManivannanMurugavel/Yolo-Annotation-Tool-New-
    https://medium.com/@manivannan_data/yolo-annotation-tool-new-18c7847a2186
    https://medium.com/@manivannan_data/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2
    https://github.com/Ma-Dan/YOLOv3-CoreML
    https://www.cognizantsoftvision.com/blog/face-detection-vision-core-image-opencv/
    https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset/downloads/cats-and-dogs-breeds-classification-oxford-dataset.zip/1
    http://vision.stanford.edu/aditya86/ImageNetDogs/


