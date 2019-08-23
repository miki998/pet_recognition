import os,sys

def verify(path_to_folder):
    # simple checking
    print("Warning: we assume that in all the files the names for them do not include dots, you can use another script that changes all the images to standardize name for it to not create any problem")
    
    
    #extension checking
    for f in os.listdir(path_to_folder):
        ext = f.split('.')[-1]
        if ext != 'jpg' and ext != 'txt':
            print('extension that does not belong to this folder')
            sys.exit()
            return

    #matching checking
    array = os.listdir(path_to_folder)
    jpgs = [f.split('.')[0] for f in array if f.split('.')[-1] == 'jpg']
    txts = [f.split('.')[0] for f in array if f.split('.')[-1] == 'txt']

    if len(jpgs) != len(txts):
        print('There are some files missing, please double check')
        sys.exit()
        return
    for x in jpgs:
        if x not in txts:
            print('There are some files missing, please double check')
            sys.exit()
            return

    print('All good, the verification done, should be fine for augmentation')



def main():
    array = sys.argv[1:]
    if len(array) == 0:
        print('Not the right number of arguments, use -h for usage recommendations')
        return
    elif array[0] == '-h':
        if len(array) > 1: print('Not the right number of arguments, for this feature, only -h args is accepted')
        else: print('-h: for helper ; -v path_to_file: for verify of a specific folder ')
    elif array[0] == '-v':
        if len(array) > 2: print('Not the right number of arguments, use -h for usage recommendations')
        elif not os.path.exists(array[1]) : print('Please enter the path to file, use -h for usage recommandations')
        elif len(array) == 2:
            if not os.path.exists(array[1]) : print('Please enter the path to file, use -h for usage recommandations')
            else: verify(array[1])
        else: print('Not the right number of arguments, use -h for usage recommendations')
    else: print('Not correct usage, use -h for usage recommendation' )



if __name__ == '__main__':
    main()
