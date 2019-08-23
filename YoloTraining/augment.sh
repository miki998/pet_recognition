#!/bin/bash

#augmentation script to write images for yolo training weights

echo 'Just as a reminder, you must have initial data available in Images folder i.e txt files and jpg files'



cd testing
rm * 2> /dev/null 
cd ../
rm *.jpg 2> /dev/null 
echo 'Cleaning the testing directory before starting'
echo 'Do not put jpg files in the darknet folder'

echo 'Running the augmentation file'
echo 'AND wait the pid to finish'
python3 augmentation.py
PID=$!
echo $PID
wait $PID

echo 'Small reorganizing job'
cd testing
mv *.jpg ../ 2> /dev/null 
mv *.txt ../ 2> /dev/null 
cd ../
mv *.jpg Images 2> /dev/null 
mv *.txt Images 2> /dev/null 

cd Images
mv *.jpg full_catAndDog 2> /dev/null 
mv *.txt full_catAndDog 2> /dev/null 


echo 'All good now, you can check the txt files and jpg files in Images/catAndDog folder'
echo 'Also if you took the verbose option, the pictures with rectangle are in the testing folder'
echo 'Miaou'




