import sys
import shutil
from pathlib import Path 

def preprocess_emoji(dset_path):
    # make emoji unicode vocabulary 
    code_vocaburary = {}
    code_path = Path('./emoji/description/unicode.txt')
    code_list = code_path.read_text(encoding='utf-8').split('\n')
    for index, data in enumerate(code_list):
        code_vocaburary[data] = index
    
    # chack dataset path
    image_path = Path(dset_path)
    if image_path.exists() == False:
        exit('Check your dataset path!')

    # copy designated emoji images
    for filepath in list(image_path.glob("*.png")):
        if str(filepath.name.split(".")[0]) in code_list:
            shutil.copyfile(filepath, \
                './emoji/edited/' + str(code_vocaburary[filepath.name.split(".")[0]]) + '.png')
                
if __name__ == '__main__':
    if len(sys.argv) == 2:
        print('Preprocessing emoji dataset...')
        preprocess_emoji(sys.argv[1])
        print('Done!')
    else:
        exit('Check your input arguments!')
