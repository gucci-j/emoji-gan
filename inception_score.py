from pathlib import Path
import sys

from cgan_emoji import DCGAN
# from utils.dataset_utils import load_dataset as load_dataset_gan

import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from PIL import Image as pil_image

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from nltk.tokenize import sent_tokenize
import re

import pandas as pd

# Reference: http://bluewidz.blogspot.com/2017/12/inception-score.html
model = InceptionV3() # Load a model and its weights

def load_dataset_gan(img_path, txt_path, img_shape=(64, 64, 3), split_rate=0.1):
	"""
	Function: load_dataset  

	Input: img_path: image directory path, 
	txt_path: caption directory path, 
	img_shape: image shape array  

	Output: NumPy array
	"""

	print('Acquiring images & labels...')

	images = dict()
	texts = dict()
	
	t_path = Path(txt_path)
	i_path = Path(img_path)

	for filename in list(i_path.glob("*.png")):
		name = filename.name.replace('.png', '')
		images[name] = filename.resolve()
		
	for filename in list(t_path.glob("*.txt")):
		name = filename.name.replace('.txt', '')
		texts[name] = filename.read_text(encoding='utf-8').lower()
	
	image_list = []
	caption_list = []
	numbers = []
	
	for name, item_path in images.items():
		if name in texts:
			text = texts[name]
			text = text.replace("“", "")
			text = text.replace("”", "")
			tokenized = sent_tokenize(text)
			label_number = int(name)

			for sentence in tokenized:
				filtered_sentence = re.sub(re.compile("[!-/:-@[-`{-~]"), "", sentence)
				image = img_to_array(load_img(item_path, target_size=(img_shape[0], img_shape[1])))
				# image = (image.astype(np.float32) / 127.5) - 1.
				image_list.append(image)
				caption_list.append(filtered_sentence)
				numbers.append(label_number)
	
	print('Done!')
	"""
	for image, caption in result:
		print(caption)
	"""
	image_list = np.array(image_list)
	caption_list = np.array(caption_list)
	numbers = np.array(numbers)
	print('>>> Dataset Size: %s' % len(image_list))
	image_train, image_test, caption_train, caption_test, numbers_train, numbers_test = train_test_split(image_list, caption_list, numbers, test_size=split_rate)

	return image_train, caption_train, image_test, caption_test, numbers_train

def inception_score(x, batch_size=26):
    r = None
    n_batch = (x.shape[0] + batch_size - 1) // batch_size
    for j in range(n_batch):
        x_batch = x[j * batch_size:(j + 1) * batch_size, :, :, :]
        r_batch = model.predict(preprocess_input(x_batch)) # r has the probabilities for all classes
        r = r_batch if r is None else np.concatenate([r, r_batch], axis=0)
    p_y = np.mean(r, axis=0) # p(y)
    e = r * np.log(r / p_y) # p(y|x)log(P(y|x)/P(y))
    e = np.sum(e, axis=1) # KL(x) = Σ_y p(y|x)log(P(y|x)/P(y))
    e = np.mean(e, axis=0)
    return np.exp(e) # Inception score

def output_inception_gan(gen_path, dis_path, X_train, Captions):
    dcgan = DCGAN(img_path, txt_path, glove_path)
    print('Loading model...')
    dcgan.load_model(gen_path, dis_path)

    _images = []
    print('Generating images...')
    for image, caption in zip(X_train, Captions):
        generated_image = dcgan.generate_image_from_text(caption,flag=False)
        _images.append(generated_image)
        generated_image = generated_image * 127.5 + 127.5
        _images.append(generated_image)
    
    return inception_score(resize(np.asarray(_images)))
    
def resize(x):
    x_list = []
    for i in range(x.shape[0]):
        img = image.array_to_img(x[i, :, :, :].reshape(64, 64, -1))
        img = img.resize(size=(299, 299), resample=pil_image.LANCZOS)
        x_list.append(image.img_to_array(img))
    return np.array(x_list).astype('float32') / 127.5

def dataset_inception(x_train):
    return inception_score(resize(x_train))

if __name__ == '__main__':
    scores = []
    img_size = (64, 64, 3)
    img_path = './emoji/edited/emoji_64x64/'
    txt_path = './emoji/description/detailed'
    glove_path = './utils/glove.6B.300d.txt'

    if len(sys.argv) == 3:
        # emoji gan
        x_train, y_train, _, _, _ = load_dataset_gan(img_path, txt_path, img_size, split_rate=0.0)
        print('Complete data loading!')

        original_score = dataset_inception(x_train)
        generated_score = output_inception_gan(sys.argv[1], sys.argv[2], x_train, y_train)
        scores.append([original_score, generated_score])
        print('Dataset score: {0}, Generated score: {1}'.format(original_score, generated_score))
        
        df = pd.DataFrame(scores, columns=['original', 'generated'])
        df.to_csv('./saved_model/inception_gan.csv', mode='a', header=False)

    else:
        exit('Invalid argument!')
