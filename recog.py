from src import *
import argparse



def main():	

	parser = argparse.ArgumentParser(description='options for training, one by one each pet')
	parser.add_argument(
		'-f', '--path_toimage', required=True, type=str, help='path to the image where you have the pet you want to recognize')
	parser.add_argument(
		'-m', '--model', required=False, type=str, help='path to previous model')
	args = parser.parse_args()

	try:
		with open('match.json','r') as fp: match = json.load(fp)
	except:
		print('No model so can\' recognize anything')
		return 

	yolob = Yolobody()
	yoloe = Yoloeye()
	yolof = Yoloface()
	haar = HaarCascadeExt()
	face_database = Model(args.model)

	im = cv2.imread(args.path_toimage)
	ret = yolob.detect(im)
	if not len(ret): print('We do not detect neither dog or cat in the {} image'.format(idx)) ; return 
	if len(ret) > 1: print('We detected more than one dog or cat in the {} image'.format(idx)) ; return 
	x,y,w,h = ret[0][2]
	X,Y,W,H = cent2rect(x,y,w,h)
	croped = im[Y:Y+H,X:X+W]

	ret2 = yolof.detect(croped)
	if not len(ret2): print('We do not detect neither dog or cat face in the {} image'.format(idx)) ; return
	if len(ret2) > 1: print('We detected more than one dog or cat face in the {} image'.format(idx)) ; return 
	x,y,w,h = ret2[0][2]
	X,Y,W,H = cent2rect(x,y,w,h)
	closeup = croped[Y:Y+H,X:X+W]
	
	ret3 = haar.detect(closeup,(0,0))[0]
	if not (len(ret3)):  print('We do not detect neither dog or cat face in the {} image'.format(idx)) ; return
	#if len(ret3) > 1: print('We detected more than one dog or cat face in the {} image'.format(idx)) ; continue
	nX,nY,nW,nH = ret3

	label,conf = face_database.prediction(closeup[nY:nY+nH,nX:nX+nW])

	print("We recognize (among all the pet) that {} looks the most like the one on the picture".format(match[int(label)]))

if __name__ == '__main__':
	main()