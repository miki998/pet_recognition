# General Idea/ Modifications

Again a sort of easy script for you to train simply yolo weight. Again run train and see helper functions. The images the you processed in advance using YOLO ANNOTATIONS TOOL or with already
prelabelled data must be put inside a folder named IMAGES itself inside DARK.

## Structure

- augmentation.py: script for augmentation of data
- augment.sh: script for above
- bin_can: useful funcs
- DARK: folder for detection
- testing: for verbose folder
- train: main script

## Usage
Note: You must have all the images you want to train on prepared inside DARK/Images with their respective annotations texts
	
	./train pathToStartingWeight
