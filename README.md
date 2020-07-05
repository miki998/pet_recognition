# General Idea/ Modifications

This is a combination of Haarcascade and yolov. As we know, we need some sort of intermediary input for cropped faces, even though those are not very much precise due to lbph's nature (background and fur influe quite a lot on its prediction). 
The general idea would be to use newly trained models (body and face) to capture... body and then face. The face would go through a cascade to be then predicted or fed to train. 
We use for now a memory determination way to decide what we should label on the detection. The decision is made on a simple average (even though we should definitely improve this). 

# Structure of the Repository

- DARK: A folder containing a modified/replaced version of a cloned library
- __version__: previous versions of the main code, you can check it out as a research material
- bin_can: some auxiliary scripts, might be useful in some cases, read the description the code itself
- models: recognition models, the one in the current repository was trained on different pets listed in PetName.txt
- video: folder where you store the videos that tests the recognition of the pets on the video
- video_result: folder where you get the outputted result of the script when given a video
- weights: the weights for the detection part
- PetName.txt: list of the pet names that our model classifies on
- cascadaYolo: bash script for a easier usage
- cascadaYolo_v5.py: python script source code for cascadaYolo


## Usage

### Preparations
Since gitlab limits the amount of files we can upload, we need to fetch back the needed weight and libs. 
If you wanna get the video yourself then go ahead, but you can also follow the third part of this preparation
in order to get the video that we already prepared for you.

- Dowload the following [files](https://drive.google.com/open?id=113ri7b58DAMj-g_ep4HAVRhv5xPfCddi) and put them inside the weights folder  
- (Only if you don't have anything inside DARK) Download the following files and replace DARK [folder](https://drive.google.com/open?id=1WynGkBqTeMHpHLSleJBB49kGNy1P2Xyx)
- (Optional) Download the [videos](https://drive.google.com/open?id=1UmLqGnGD6Vv0jaLtKc60YGzrnZKofeTJ) and put them in video folder.
- (Optional) Download the [video_result](https://drive.google.com/open?id=1PJRPwJXsXJ_8-euIUf1M_8q4sb1IwJkK) , these are some result of our recognition. (the files might not be up to date)

You should be good to go now ... as if xD. If you are not using a GPU then while running the DARK scripts you will
encounter multiple errors. So either get yourself a GPU (would be nice) or :
- go to DARK folder you should find a Makefile
- Check the source code of Makefile, and change the first line to GPU = 0
- Save, make, now good to go.
    
### Utility
Check the helper function yourself

    ./cascadaYolo -h
    
or if you are lazy then here are the different possibilities (videopath, is the path to the video from root of this repository/can be absolute path too if you wish): 
    
    ./cascadaYolo -r videopath
    ./cascadaYolo -r -v videopath
    ./cascadaYolo -l 
    ./cascadaYolo -l -v 
    
- the -r option is for recognition of a video
- the -l option is for live recognition but you'll need a webcam with index 0 in the list of your cameras.
- the -v option is basically verbose i.e having the bounding for yoloface (red rectangle) and the bounding for haarcascade that is fed to predict (blue rectangle)

When using -r, you should get the output in video_result, with the name r"videoname". 
