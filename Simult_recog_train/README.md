## General Idea/Modifications

This time around, the biggest difference is that we change the model as we go through a video or live. Exceptions might arise because of special cases. 
We will be using a slightly smarter weighted average to determine what to label on detection, a textfile is updated each time we do recognition and notice some pets that is supposedly not in our database.

## Structure

- DARK: folder for detection
- models: models for recognition
- PetName.txt: the pet list of pets recognized from models
- recogAndTrain: main script 
- recogAndTrain.py: main script
- testing: folder to store some verbose/debug pics
- video: video testing dataset
- video_result: Output of the video as Input
- weight: weights for detection
- __version__: previous versions

## Usage

Use helper func -h

	./recogAndTrain -h
or 
	
	./recogAndTrain -r
	./recogAndTrain -r -v
	./recogAndTrain -l
	./recogAndTrain -l -v

