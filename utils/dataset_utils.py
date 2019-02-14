# coding: utf-8

from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from nltk.tokenize import sent_tokenize
import numpy as np
import sys
import re

def load_dataset(img_path, txt_path, img_shape=(64, 64, 3), split_rate=0.1):
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
				image = (image.astype(np.float32) / 127.5) - 1.
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
    
if __name__ == '__main__':
	img_size = (64, 64, 3)
	img_path = '../emoji/edited/emoji_64x64/'
	txt_path = '../emoji/description/detailed/'
	load_dataset(img_path, txt_path, img_size)
