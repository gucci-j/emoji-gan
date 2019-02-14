from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Lambda
from utils.glove_loader import GloveModel
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils.dataset_utils import load_dataset
from PIL import Image
import math
import pandas as pd
import sys
import time

# GPU setting
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(
                visible_device_list="2", # specify GPU number
                allow_growth=True)
        )
set_session(tf.Session(config=config))


class DCGAN():
    def __init__(self, img_path, txt_path, glove_path):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.embedding_dim = 300
        self.img_path = img_path
        self.txt_path = txt_path
        self.glove_path = glove_path

        optimizer_g = Adam(0.0005, 0.5)
        optimizer_d = Adam(0.00005, 0.5)

        # Build the GloVe model
        self.glove_model = GloveModel()
        self.glove_model.load(data_dir_path=self.glove_path, embedding_dim=self.embedding_dim)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_d,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        cond_input = Input(shape=(self.embedding_dim,))
        img = self.generator([z, cond_input])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator([img, cond_input])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z, cond_input], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_g)

    def build_generator(self):
        generator_input = Input(shape=(self.latent_dim, ), name="g_input")
        cond_input = Input(shape=(self.embedding_dim, ), name="cond_g_input")
        cond_output = Dense(100)(cond_input)

        G = concatenate([generator_input, cond_output])
        G = Dense(256 * 8 * 8, activation="relu")(G)
        G = Reshape((8, 8, 256))(G)
        G = UpSampling2D()(G)
        G = Conv2D(256, kernel_size=3, padding="same")(G)
        G = BatchNormalization(momentum=0.8)(G)
        G = Activation("relu")(G)
        G = UpSampling2D()(G)
        G = Conv2D(128, kernel_size=3, padding="same")(G)
        G = BatchNormalization(momentum=0.8)(G)
        G = Activation("relu")(G)
        G = UpSampling2D()(G)
        G = Conv2D(64, kernel_size=3, padding="same")(G)
        G = BatchNormalization(momentum=0.8)(G)
        G = Activation("relu")(G)
        G = Conv2D(self.channels, kernel_size=3, padding="same")(G)
        generator_output = Activation("tanh")(G)

        generator = Model([generator_input, cond_input], generator_output)
        generator.summary()

        return generator

    def build_discriminator(self):
        discriminator_input = Input(shape=self.img_shape, name="d_input")
        cond_input = Input(shape=(self.embedding_dim, ), name="cond_d_input")
        D = Conv2D(64, kernel_size=3, strides=2, padding="same")(discriminator_input)
        D = LeakyReLU(alpha=0.2)(D)
        D = Dropout(0.25)(D)
        D = Conv2D(128, kernel_size=3, strides=2, padding="same")(D)
        D = ZeroPadding2D(padding=((0,1),(0,1)))(D)
        D = BatchNormalization(momentum=0.8)(D)
        D = LeakyReLU(alpha=0.2)(D)
        D = Dropout(0.25)(D)
        D = Conv2D(256, kernel_size=3, strides=1, padding="same")(D)
        D = BatchNormalization(momentum=0.8)(D)
        D = LeakyReLU(alpha=0.2)(D)
        D = Dropout(0.25)(D)
        D = Conv2D(512, kernel_size=3, strides=2, padding="same")(D)
        D = BatchNormalization(momentum=0.8)(D)
        D = LeakyReLU(alpha=0.2)(D)

        cond_d_hidden = Dense(100)(cond_input)
        cond_d_hidden = Reshape((1, 1, 100))(cond_d_hidden)
        cond_d_output = Lambda(lambda x: K.tile(x, [1, 9, 9, 1]))(cond_d_hidden)

        D = concatenate([D, cond_d_output], axis=-1)
        D = Conv2D(512, kernel_size=3, strides=1, padding='same')(D)
        D = BatchNormalization(momentum=0.8)(D)
        D = LeakyReLU(alpha=0.1)(D)
        D = Dropout(0.25)(D)
        D = Flatten()(D)
        discriminator_output = Dense(1, activation='sigmoid')(D)

        discriminator = Model([discriminator_input, cond_input], discriminator_output)
        discriminator.summary()

        return discriminator

    def train(self, epochs, batch_size=26, save_interval=20):
        # load dataset
        X_train, Captions, X_test, Captions_test, Labels = load_dataset(self.img_path, self.txt_path, self.img_shape)
        caption_list_train = []
        caption_list_test = []
        for caption in Captions:
            caption_list_train.append([str(caption)])
        for caption in Captions_test:
            caption_list_test.append([str(caption)])
        df = pd.DataFrame(caption_list_train, columns=['caption'])
        df.to_csv('./saved_model/caption_train.csv')
        df = pd.DataFrame(caption_list_test, columns=['caption'])
        df.to_csv('./saved_model/caption_test.csv')

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        batch_count = int(X_train.shape[0] / batch_size)
        history = []
        history_test = []

        for epoch in range(epochs):
            for batch_index in range(batch_count):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half of images
                # idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[batch_index * batch_size:(batch_index + 1) * batch_size]
                texts_input = Captions[batch_index * batch_size:(batch_index + 1) * batch_size]
                texts = self.glove_model.encode_docs(texts_input)

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict([noise, texts])

                # Train the discriminator (real classified as ones and generated as zeros)
                start = time.time()
                d_loss_real = self.discriminator.train_on_batch([imgs, texts], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, texts], fake)
                batch_time_d = time.time() - start
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (wants discriminator to mistake images as real)
                start = time.time()
                g_loss = self.combined.train_on_batch([noise, texts], valid)
                batch_time_g = time.time() - start

                # Plot the progress
                batch_time = batch_time_d + batch_time_g
                print ("%d-%d [D loss: %f, acc.: %.2f%%] [G loss: %f] [Time: %f]" % (epoch, batch_index, d_loss[0], 100*d_loss[1], g_loss, batch_time))
                history.append([epoch, batch_index, d_loss[0], 100*d_loss[1], g_loss, batch_time])
            
            # Test the model
            texts_test = self.glove_model.encode_docs(Captions_test)
            noise_test = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs_test = self.generator.predict([noise_test, texts_test])
            start = time.time()
            d_loss_real_test = self.discriminator.test_on_batch([X_test, texts_test], valid)
            d_loss_fake_test = self.discriminator.test_on_batch([gen_imgs_test, texts_test], fake)
            batch_time_d_test = time.time() - start
            d_loss_test = 0.5 * np.add(d_loss_real_test, d_loss_fake_test)
            start = time.time()
            g_loss_test = self.combined.test_on_batch([noise_test, texts_test], valid)
            batch_time_g_test = time.time() - start

            # Plot the test progress
            batch_time_test = batch_time_d_test + batch_time_g_test
            print ("%d (test) [D loss: %f, acc.: %.2f%%] [G loss: %f] [Time: %f]" % (epoch, d_loss_test[0], 100*d_loss_test[1], g_loss_test, batch_time_test))
            history_test.append([epoch, d_loss_test[0], 100*d_loss_test[1], g_loss_test, batch_time_test])

            # If at save interval => save generated image samples & training weights
            if epoch % save_interval == 0:
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                texts_input = Captions[idx]
                texts = self.glove_model.encode_docs(texts_input)
                self.save_imgs(epoch, texts)

                self.generator.save_weights(filepath='./saved_model/generator_weights_' + str(epoch) + '.h5')
                self.discriminator.save_weights(filepath='./saved_model/discriminator_weights_' + str(epoch) + '.h5')
        
        # save weights & history
        df_train = pd.DataFrame(history, columns=['epoch', 'batch', 'd_loss', 'acc', 'g_loss', 'time[sec]'])
        df_train.to_csv('./saved_model/history.csv')
        df_test = pd.DataFrame(history_test, columns=['epoch', 'd_loss', 'acc', 'g_loss', 'time[sec]'])
        df_test.to_csv('./saved_model/history_test.csv')
        self.generator.save_weights(filepath='./saved_model/generator_weights.h5')
        self.discriminator.save_weights(filepath='./saved_model/discriminator_weights.h5')

    def save_imgs(self, epoch, texts, batch_size=26):
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        if batch_size == 260:
            texts = self.glove_model.encode_docs(texts)
        gen_imgs = self.generator.predict([noise, texts])
        gen_img = combine_normalized_images(gen_imgs)
        img_from_normalized_img(gen_img).save("images/snapshot/%d.png" % epoch)
    
    def load_model(self, gen_path='./saved_model/generator_weights.h5', dis_path='./saved_model/discriminator_weights.h5'):
        """
        Function: load_model  
        This function loads a pre-trained model.  

        Input: model_dir_path: designate where weights file is.  
        Output: None (pre-trained model will be loaded.)
        """

        ### load weights
        self.generator.load_weights(gen_path)
        self.discriminator.load_weights(dis_path)
    
    def generate_image_from_text(self, text, flag=True):
        ### prepare an empty array
        noise = np.zeros(shape=(1, self.latent_dim))
        encoded_text = np.zeros(shape=(1, self.embedding_dim))

        ### generate sample for input data
        encoded_text[0, :] = self.glove_model.encode_doc(text)
        noise[0, :] = np.random.uniform(0, 1, self.latent_dim)

        ### predict and generate an image
        generated_images = self.generator.predict([noise, encoded_text])
        generated_image = generated_images[0]

        if flag is True:
            generated_image = generated_image * 127.5 + 127.5
            return Image.fromarray(generated_image.astype(np.uint8))
        elif flag is not True:
            return generated_image

def combine_normalized_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img
    return image

def img_from_normalized_img(normalized_img):
    image = normalized_img * 127.5 + 127.5
    return Image.fromarray(image.astype(np.uint8))

def generate_mode():
    img_size = (64, 64, 3)
    img_path = './emoji/edited/emoji_64x64/'
    txt_path = './emoji/description/detailed'
    glove_path = './utils/glove.6B.300d.txt'

    dcgan = DCGAN(img_path, txt_path, glove_path)
    X_train, Captions, _, _, _ = load_dataset(img_path, txt_path, img_size, split_rate=0.0)
    print('Loading model...')
    dcgan.load_model()

    iteration = 0
    caption_list = []
    print('Generating images...')
    for image, caption in zip(X_train, Captions):
        edited_image = image * 127.5 + 127.5
        edited_image = Image.fromarray(edited_image.astype(np.uint8))
        edited_image.save('./images/original/' + str(iteration) + '.png')
        generated_image = dcgan.generate_image_from_text(caption)
        generated_image.save('./images/output/' + str(iteration) + '.png')
        caption_list.append([str(caption)])
        iteration += 1

    df = pd.DataFrame(caption_list, columns=['caption'])
    df.to_csv('./images/caption.csv')

    # plot all emojis
    dcgan.save_imgs(epoch=5000, texts=Captions, batch_size=260)
    print('Done!')

def train_mode():
    img_path = './emoji/edited/emoji_64x64/'
    txt_path = './emoji/description/detailed'
    glove_path = './utils/glove.6B.300d.txt'

    dcgan = DCGAN(img_path, txt_path, glove_path)
    dcgan.train(epochs=5000, batch_size=26, save_interval=50)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == '1':
            generate_mode()
        elif sys.argv[1] == '0':
            train_mode()
        else:
            print("Unexpected Input Value!")
    
