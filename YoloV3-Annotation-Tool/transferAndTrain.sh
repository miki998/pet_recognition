#!/bin/bash

#TODO Didn't really have time to modify paths after structure change of the whole project, so... not tested, this script has more a informative value, to tell you how to use the project and do the following crop -> bbox -> train -> detect


#Purpose: for direct use of annoted images with its outputed label, so right after using main.py, and then immediatly kicks the training process for darknet on a new empty weight -> darknet54conv (you can change this default option here)

#This is somehow custom made for folders that we are using right now for cat and dog recog, please do the necessary change if you are doing it on something else

#includes the following steps -> process.py/verify.py/augment.sh/detector train


weight = 'weights/darknet53.conv.74'
nb_files = ls Images/full_catAndDog/ | wc -l
if [ $nb_files == 0 ]
then 
	echo "You don't have any picutres in the Images/full_catAndDog/ folder \n"
else
	cp Labels/full_catAndDog/* Images/full_catAndDog	
fi

echo "We now verify whether the folder is ready for augmentation \n\n"

python3 verify.py '-v' 'Images/full_catAndDog'
python3 process.py


echo "copying images and its labels to the folder where we train \n"

cp -r Images/ ../Method_CascadaYolo/
cp cat-dog-test.txt ../Method_CascadaYolo/order/
cp cat-dog-train.txt ../Method_CascadaYolo/order/

cd ../Method_CascadaYolo/
echo "Time to augment the data \n"
./augment.sh


echo "READY ! Start training, take a sip of coffee sleep a bit, and come back tomorrow ... \n"
./darknet detector train "cfg/cat-dog-obj.data" "cfg/cat-dog-yolov3-tiny.cfg" $weight


