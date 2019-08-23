#!/bin/bash

#used for getting the detections or the recognition for a whole folder named video, and it outputs the videos results in the folder video_result

if [ $1 == "-m" ]
then
	for file in video/*;
		do python3 cascadaYolo_v4.py $1 "$file";
	done
elif [ $1 == "-r" ]
then 
	if [ $2 == "-v" ]
	then 
		for file in video/*;
			do python3 cascadaYolo_v4.py $1 $2 "$file";
		done
	else 
		for file in video/*;
			do python3 cascadaYolo_v4.py $1 "$file";
		done
	fi
else
	echo "Wrong usage: only -m or -r and -v"
fi
