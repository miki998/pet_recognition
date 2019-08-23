# YoloV3-Annotation-Tool


The goal is to create text files in darknet yolo format that will be fed to training to the CNN. 
Below is explained pretty much how to use the functionalities. 
Disclaimer: Credits to ManivannanMurugavel/, you can find his repo here [documentations](https://github.com/ManivannanMurugavel/Yolo-Annotation-Tool-New-) . This repos is somewhat a correction of his repo, as being busy probably did he not notice the issue I raised about not displaying the whole picture. 


## General Prerequisites
    python3
    tkinter
    (From the commands it is implicit that this is exact commands are targetting Mac OSX and Linux, you will have to tweak by yourself for Windows, or ask me questions)

## Structure:
    Images: folder where you have to throw in the pictures you want to annotate
    Labels: automatically a respective yolo format self of the images' in the folder Images will be created in Labels with the same path. 
    classes.txt: the different class of objects you want to detect using yolov3, write in the same syntax as in the one cloned directly from our repo.
    main.py: GUI for annotations
    process.py: creates txt files in yolov3 format
    train.txt/test.txt: (or names similar to it, created from process.py) txt files in yolov3 format to be used during training too
## Usage:
In order to use the GUI and then create the files for the "orders" that receives the training for darknet:
On terminal do the following: 
```
python3 main.py
python3 process.py -arg
```

To know usage of process.py, to "-h" command.

## EXTRA <Please check out my other repo for full applications>
Enjoy ! And look at my other repositories, especially this one : [my_repo](https://github.com/aka9/cat_dog_recog) where I have a full applications of detection using yoloV3 and haarcascade (using Custom Data) and recognition for cats and dogs ! (demo with camera possible or IOS Apps)


