# General Idea/ Modifications

demo_detct is a folder used to display the use of yolo for detection. For usage you simply have to add the images in the imagedetection folder, and the weights you wanna test out in the weight folder. The usage of the detect script is detailed when used with -h helper.

## Structure

- DARK: Folder for detection use
- detect: bash script to use
- imageDetectionTest: self explanatory
- result: output of detection with bounding box drawn
- weight: weight for detection that you can choose from

## Usage

Simply run detect script, -h helper to know what to do

	./detect -h
	./detect pathToImage pathToWeight
