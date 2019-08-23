## General Idea/Modifications

This folder has scripts that helps detecting some pet features, for now only eyes. We then use this eye detection to do some rotations to have a horizontal sort of face.
Horizontalize bash script with -h helper for more detail. 
Would require some more features for it to be able to do align of a profile face. 

## Structure

- DARK: folder for detection
- result: result of the horizontalization, we should see a rotated picture
- test_images: the database of images for testing horizontalization
- verbose: show where the bboxes for eyes are
- horizontalize.py: main python script
- horizontalize: bash script for horizontalization of a picture

## Usage

The main script horizontalize requires the path to an image, and will output either a horizontalized photo in result (along eyes) or will raise an error if eyes are not detected

	./horizontalize pathToImage
