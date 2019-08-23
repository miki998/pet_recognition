import glob, os, sys



def process(current_dir):
    #current_dir = 'Images/full_catAndDog'

    # Percentage of images to be used for the test set
    percentage_test = 10;

    # Create and/or truncate train.txt and test.txt
    file_train = open('textes/cat-dog-train.txt', 'w')
    file_test = open('textes/cat-dog-test.txt', 'w')

    # Populate train.txt and test.txt
    counter = 1
    index_test = round(100 / percentage_test)
    print("Warning: we assume that all pictures to be processed are jpgs")
    for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test:
            counter = 1
            file_test.write(current_dir + "/" + title + '.jpg' + "\n")
        else:
            file_train.write(current_dir + "/" + title + '.jpg' + "\n")
            counter = counter + 1

def main():
    array = sys.argv[1:]
    if len(array) > 1: print("Not the right number of arguments, use -h for usage recommendations")
    elif len(array) == 0: print("Not the right number of arguments, use -h for usage recommendations")
    elif array[0] == '-h': print('simply write the path_to_folder as an argument')
    else:
        if not os.path.exists(array[0]): print('folder not found')
        else: process(array[0])
        


if __name__ == '__main__':
    main()