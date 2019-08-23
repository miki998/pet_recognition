## General Idea/Modifications

This folder has scripts that helps visualizing lbp features of a face, simply use faceRepresentation script if you wanna try to have the lbp of a cropped face (do not get output every time since
for some pictures the face is not detected) else you can just use representation for lbp representation of the picture as it is.

## Structure

- DARK: folder for detection
- specimenVerbose: just a random image database
- test_images: same as specimenVerbose
- faceRepresentation.py: script that takes in any sort of image and try to give you the LBP of the face it founds in the image
- faceRepresentation: script for above
- representation.py: script that takes in an image and gives back the image's LBP
- representation: script for above

## Usage
Usage -h helper if needed:
	
	./faceRepresentation pathToImage
	./representation pathToImage

