import os,sys

#name standardizing

def standardize(path_to_folder):
    # simple checking
    print("Warning: Do this before the process.py and before verify.py, technically, use verify.py to check the matching too, but it is fine anyways")
    

    #matching checking
    array = os.listdir(path_to_folder)
    jpgs = [f for f in array if f.split('.')[-1] == 'jpg']
    txts = [f[:-4] for f in array if f.split('.')[-1] == 'txt']
    
    number = 0
    
    
    for file in jpgs:
        filename = file[:-4]
        if filename in txts:
            os.rename(path_to_folder+'/'+file,path_to_folder+'/pet'+str(number)+'.jpg')
            os.rename(path_to_folder+'/'+filename+'.txt',path_to_folder+'/pet'+str(number)+'.txt')
        number += 1
    
    print('All good, names should be fine')



def main():
    array = sys.argv[1:]
    if len(array) == 0:
        print('Not the right number of arguments, use -h for usage recommendations')
        return
    elif array[0] == '-h':
        if len(array) > 1: print('Not the right number of arguments, for this feature, only -h args is accepted')
        else: print('-h: for helper ; -s path_to_file: for standardizing of a specific folder ')
    elif array[0] == '-s':
        if len(array) > 2: print('Not the right number of arguments, use -h for usage recommendations')
        elif not os.path.exists(array[1]) : print('Please enter the path to file, use -h for usage recommandations')
        elif len(array) == 2:
            if not os.path.exists(array[1]) : print('Please enter the path to file, use -h for usage recommandations')
            else: standardize(array[1])
        else: print('Not the right number of arguments, use -h for usage recommendations')
    else: print('Not correct usage, use -h for usage recommendation' )



if __name__ == '__main__':
    main()




