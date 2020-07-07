from src import *
import argparse



def main():

	parser = argparse.ArgumentParser(description='options for training, one by one each pet')
	parser.add_argument(
		'-f', '--folder', required=True, type=str, help='path to the folder image where you have the pet you want to recognize')
	parser.add_argument(
		'-n', '--name', required=True, type=str, help='pet\'s name')
	parser.add_argument(
		'-m', '--model', required=False, type=str, help='path to previous model')
	args = parser.parse_args()
	

	yolob = Yolobody()
	yoloe = Yoloeye()
	yolof = Yoloface()
	haar = HaarCascadeExt()
	face_database = Model(args.model)

	try:
		with open('match.json','r') as fp: match = json.load(fp)
	except:
		match = {}
	next_idx = len(match)

	succ = 0
	specimen = []
	for idx,im_path in enumerate(os.listdir(args.folder),start=0):
		im = cv2.imread(im_path)
		ret = yolob.detect(im)
		if not len(ret): print('We do not detect neither dog or cat in the {} image'.format(idx)) ; continue
		if len(ret) > 1: print('We detected more than one dog or cat in the {} image'.format(idx)) ; continue
		x,y,w,h = ret[0][2]
		X,Y,W,H = cent2rect(x,y,w,h)
		croped = im[Y:Y+H,X:X+W]

		ret2 = yolof.detect(croped)
		if not len(ret2): print('We do not detect neither dog or cat face in the {} image'.format(idx)) ; continue
		if len(ret2) > 1: print('We detected more than one dog or cat face in the {} image'.format(idx)) ; continue
		x,y,w,h = ret2[0][2]
		X,Y,W,H = cent2rect(x,y,w,h)
		closeup = croped[Y:Y+H,X:X+W]
		
		ret3 = haar.detect(closeup,(0,0))[0]
		if not (len(ret3)):  print('We do not detect neither dog or cat face in the {} image'.format(idx)) ; continue
		#if len(ret3) > 1: print('We detected more than one dog or cat face in the {} image'.format(idx)) ; continue
		nX,nY,nW,nH = ret3
		specimen.append(closeup[nY:nY+nH,nX:nX+nW])
		succ += 1

	print('Only {} images were taken to train'.format(succ))
	if succ: 
		face_database.training(specimen,next_idx)
		match[next_idx] = args.name

		with open('match.json', 'w') as fp:
			json.dump(match, fp)
	

if __name__ == '__main__':
	main()