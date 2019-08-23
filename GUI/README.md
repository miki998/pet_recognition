# General Idea/ Modifications


Alright this is a MUST use GUI, cause it is useful for you to get good crops of the pet you wanna recognize. We will use some eye detection arguments to give the user
feedback on whether the position the pet is in now is good for crop or not, if not the GUI won't save it, else it will. Up to 5 pictures, (for now we decide 5). The output of the process
should be a good non redundant data for recognition training.

## Structure

- DARK: folder for prediction
- models: folder where you store the models you trained or pre-trained
- guiV2.py: current version of the GUI
- PetName.txt: the database of Pet names that we cna recognize
- sample: folder storing the cropped picture when using GUI
- __version__: previous versions of the GU


## Usage

    python3 guiV2.py
    HAVE A WEBCAMMM set to 0 idx

There should be a notification for you when the 5 crops are taken. The GUI will then automatically quit and you should find your cropped in sample.
