from keras import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from nltk.tokenize import sent_tokenize
import numpy as np
import sys
import re

from keras.utils import np_utils
from keras.callbacks import CSVLogger, EarlyStopping

from cgan_emoji import DCGAN
from utils.dataset_utils import load_dataset as load_dataset_gan

from PIL import Image
import pandas as pd

def load_dataset(img_path, txt_path, img_shape=(64, 64, 3), split_rate=0.1):
    """
	Function: load_dataset  

	Input: img_path: image directory path, txt_path: caption directory path, 
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
    
    _images = []
    _labels = []
    for name, item_path in images.items():
        if name in texts:
            text = texts[name]
            text = text.replace("“", "")
            text = text.replace("”", "")
            tokenized = sent_tokenize(text)
            for sentence in tokenized:
                image = img_to_array(load_img(item_path, target_size=(img_shape[0], img_shape[1])))
                image = (image.astype(np.float32) / 255) * 2 - 1
                _images.append(image)
                _labels.append(int(name))
    
    _images = np.array(_images)
    _labels = np.array(_labels)
    print('>>> Dataset Size: %s' % len(_images))
    print('>>> Dataset Size: %s' % len(_labels))
    (x_train, x_test, y_train, y_test) = train_test_split(_images, _labels, test_size=split_rate)
    y_train = np_utils.to_categorical(y_train, 82)
    y_test = np_utils.to_categorical(y_test, 82)
    print(y_train[0])

    return (x_train, y_train), (x_test, y_test)

def make_label():
    """
    Function: make_label  

    Input: None  
    Output: x_data, y_data  
    """
    img_size = (64, 64, 3)
    img_path = './emoji/edited/emoji_64x64'
    txt_path = './emoji/description/detailed/'
    return load_dataset(img_path, txt_path, img_size)

def build_classifier(img_shape, num_classes):
    """
    Function: build_classifier  

    Input: img_shape: image shape, num_classes: the number of image classes  
    Output: CNN-based classifier model  
    """
    cinput = Input(shape=img_shape, name='classifier_input')

    H = Conv2D(64, kernel_size=(5, 5), activation='relu')(cinput)
    H = Conv2D(128, (5, 5), activation='relu')(H)
    H = MaxPooling2D(pool_size=(2, 2))(H)
    H = Dropout(0.25)(H)
    H = Flatten()(H)
    H = Dense(128, activation='relu')(H)
    H = Dropout(0.5)(H)

    coutput = Dense(num_classes, activation='softmax')(H)

    return Model(cinput, coutput)

def train():
    """
    Function: train  
    This function trains CNN-based model.  

    Input: None  
    Output: None (model weight file will be saved.)   
    """
    img_shape = (64, 64, 3)
    num_classes = 82
    epochs = 100
    model = build_classifier(img_shape, num_classes)
    model.summary()
    
    (x_train, y_train), (x_test, y_test) = make_label()

    model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

    callbacks = []
    callbacks.append(CSVLogger('./saved_model/history_classifier.csv'))
    callbacks.append(EarlyStopping(patience=2, verbose=1))

    history = model.fit(x=x_train, y=y_train, 
        batch_size=26, epochs=epochs, verbose=1,
        validation_data=(x_test, y_test), callbacks=callbacks)
    
    model.save_weights('./saved_model/classifier_weights.h5')

def emoji_gan_generator(gen_path, dis_path):
    """
    Function: emoji_vae_generator  
    This function generates images based on inpout captions with emoji vae model.  

    Input: None
    Output: output images and corresponding labels.  
    """
    img_size = (64, 64, 3)
    img_path = './emoji/edited/emoji_64x64/'
    txt_path = './emoji/description/detailed'
    glove_path = './utils/glove.6B.300d.txt'

    dcgan = DCGAN(img_path, txt_path, glove_path)
    X_train, Captions, _, _, Labels = load_dataset_gan(img_path, txt_path, img_size, split_rate=0.0)
    print('Loading model...')
    dcgan.load_model(gen_path, dis_path)

    iteration = 0
    caption_list = []
    _images = []
    _labels = []
    print('Generating images...')
    for image, caption in zip(X_train, Captions):
        edited_image = image * 127.5 + 127.5
        edited_image = Image.fromarray(edited_image.astype(np.uint8))
        edited_image.save('./images/original/' + str(iteration) + '.png')
        
        generated_image = dcgan.generate_image_from_text(caption,flag=False)
        _images.append(generated_image)
        generated_image = generated_image * 127.5 + 127.5
        generated_image = Image.fromarray(generated_image.astype(np.uint8))
        generated_image.save('./images/output/' + str(iteration) + '.png')
        caption_list.append([str(caption)])
        _labels.append(Labels[iteration])
        iteration += 1
    
    _images = np.array(_images)
    _labels = np.array(_labels)
    print('>>> Dataset Size: %s' % len(_images))
    print('>>> Dataset Size: %s' % len(_labels))
    _labels = np_utils.to_categorical(_labels, 82)

    df = pd.DataFrame(caption_list, columns=['caption'])
    df.to_csv('./images/output/caption_classifier.csv')

    return _images, _labels

def classify(classifier_weights, gen_path, dis_path):
    """
    Function: classify  
    This function classifies output images from emoji GAN by using trained CNN-based classifier.  

    Input: classifier_weights
    Output: loss and accuracy  
    """
    img_shape = (64, 64, 3)
    num_classes = 82
    model = build_classifier(img_shape, num_classes)
    model.compile(loss=categorical_crossentropy,
                optimizer=Adadelta(),
                metrics=['accuracy'])
    model.load_weights(classifier_weights)
    x_test, y_test = emoji_gan_generator(gen_path, dis_path)
    result = model.evaluate(x=x_test, y=y_test, batch_size=26)
    print('Accuracy: {0}'.format(result[1]))
    return result[0], result[1]

if __name__ == '__main__':
    loss_acc_list = []

    if sys.argv[1] == '0':
        train()
    elif sys.argv[1] == '1' and len(sys.argv) == 5:
        loss, acc = classify(sys.argv[2], sys.argv[3], sys.argv[4])
        loss_acc_list.append([loss, acc])
        df = pd.DataFrame(loss_acc_list, columns=['loss', 'acc'])
        df.to_csv('./saved_model/acc_gan.csv', mode='a', header=False)
    else:
        print('Invalid argument!')
    