![](https://img.shields.io/badge/<Implementation>-<yolo+lbph>-<success>)
![](https://img.shields.io/badge/<Implementation>-<real_time_recognition/alarm>-<success>)

[![ko-fi](https://www.ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R5R11K2H4)
# General Idea/ Modifications

This is a combination of Haarcascade and yolov. As we know, we need some sort of intermediary input for cropped faces, even though those are not very much precise due to lbph's nature (background and fur influe quite a lot on its prediction). 
The general idea would be to use newly trained models (body and face) to capture... body and then face. The face would go through a cascade to be then predicted or fed to train. 
We use for now a memory determination way to decide what we should label on the detection. The decision is made on a simple average (even though we should definitely improve this). 

## Getting Started

Simply open the jupyter notebook and see how some demo on pictures that we uploaded with this repository

### Prerequisites

What things you need to install the software and how to install them

```
scikit-image
matplotlib
numpy
notebook
scipy
```

### Installing

Here are the steps to follow

#### Usual way
Installing using requirements.txt
```
pip3 install -r requirements.txt
```

#### Docker way
Installing using docker (if you have it installed it can make sure there is no problem linked to packages in the whole process)
```
docker build -t <docker-name> .
docker run -it --ipc=host -p 9999 <docker-name> 
```


Obviously you are free to add any options, here I added 9999 port in case you want to access with a jupyter notebook, and --ipc=host in case you want to train for new models of darknet itself (though we do not support this)

## Running the tests

When you are in the root folder of the repository or when you activated docker container 

For recognition of a specific image (you can check the argument when you mistakely run at the beginning) and for training for recognition (so computing the lbph features and saving them)
```
 python3 recog.py <-arguments>
 python3 train.py <-arguments>
```

### Break down into end to end tests

We explain here sohrtly how we proceed to get the image we extract lbph features from (extra variants can be made starting from this idea)

#### A three step detection/recognition 
- body detection
- face detection
- closeup face detection
- recognition using comparison of lbph features


## Deployment

None yet, you can do some pull requests to me

## Built With

* [python3](https://www.python.org/download/releases/3.0/) - The web framework used

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors
Michael Chan
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments
https://github.com/pjreddie/darknet









