import os
import argparse
import cv2
from datetime import datetime as dt
from avians.util.decorators import static_vars
import avians.util.image as aui
import avians.nn.common as anc
from avians.nn.classes import AutoencoderDataset

# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping

import numpy as np

import logging
LOG = logging.getLogger(__name__)

def this_file(): 
    return os.path.splitext(os.path.basename(__file__))[0]

@static_vars(timestamp=dt.now())
def save_dir(root="/tmp"):
    the_dir = "{}/{}-{}/".format(root, 
                                 this_file(),
                                 save_dir.timestamp.strftime("%F-%T"))
    if not os.path.isdir(the_dir): 
        os.makedirs(the_dir)
    return the_dir

def build_autoencoder_cnn(img_rows=64,
                          img_cols=64, 
                          img_channels=1, 
                          nb_filters=32, 
                          nb_conv=2, 
                          nb_pool=2):


    input_img = Input(shape=(img_channels, img_rows, img_cols))

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    # at this point the representation is (32, img_rows/8, img_cols/8)
    x = Flatten()(x)
    x = Dense(512)(x)
    encoded = Dense(16)(x)

    x = Dense(16)(encoded)
    x = Dense(512)(x)
    x = Reshape((8, 8, 8))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(img_channels, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    encoder = Model(input=input_img, output=encoded)
    encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    encoded_shape = (8, img_rows//8, img_cols//8)
    encoded_input = Input(shape=encoded_shape)
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    decoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return (autoencoder, decoder, encoder)


def create_word_images(word, fonts=['Amiri', 'Droid Sans Arabic'], dpis=[1200]): 
    res_list = []
    for font in fonts: 
        for dpi in dpis: 
            word_img = aui.get_gray_word_image(word,
                                               fontname=font,
                                               dpi=dpi,
                                               background=0,
                                               foreground=255)
            res_list.append(word_img)

    return res_list

def train_model(dataset,
                autoencoder,
                decoder,
                encoder,
                model_key="",
                batch_size=32,
                nb_epoch=500,
                early_stop_patience=30): 

    x_train = dataset.as_dataset()
    np.random.shuffle(x_train)

    stop_early = EarlyStopping(monitor='loss', 
                               patience=early_stop_patience, 
                               verbose=1, 
                               mode='auto')

    # model_checkpoint = ModelCheckpoint(save_dir() + "checkpoint-weights-epoch-{epoch:02d}-val-loss-{val_loss:.2f}.h5",  # noqa
    #                                    monitor='val_loss', 
    #                                    verbose=1, 
    #                                    save_best_only=True, 
    #                                    mode='auto')

    print("Fitting Model: {}".format(autoencoder))
    print("Data Shape: {}".format(x_train.shape))

    val_data = anc.prepare_validation_data(x_train, x_train)

    autoencoder.fit(x_train, 
                    x_train, 
                    batch_size=batch_size, 
                    shuffle=True,
                    nb_epoch=nb_epoch,
                    verbose=1, 
                    validation_split=0.0, 
                    validation_data=val_data,
                    callbacks=[stop_early])

    anc.save_and_upload_model(autoencoder, save_dir(), "autoencoder")
    anc.save_and_upload_model(decoder, save_dir(), "decoder")
    anc.save_and_upload_model(encoder, save_dir(), "encoder")

    return autoencoder, decoder, encoder

def remove_zeros_and_reshape(component_arr): 
    "component arr is n*x*y. We convert it to n * 1 * x * y and remove all zeros"

    nonzero_elements = []
    n, sizex, sizey = component_arr.shape
    for i, img in enumerate(component_arr): 
        if np.any(img > 0): 
            nonzero_elements.append(i)
    new_arr = np.empty(shape=(len(nonzero_elements), 1, sizex, sizey),
                       dtype=component_arr.dtype)
    for ni, oi in enumerate(nonzero_elements):
        new_arr[ni, 0] = component_arr[oi]
    return new_arr


def text_detection_ac_v1(wordlist_file, size=64, **kwargs):
    autoencoder, decoder, encoder = build_autoencoder_cnn()
    print("AUTOENCODER")
    autoencoder.summary()
    print("DECODER")
    decoder.summary()
    print("ENCODER")
    encoder.summary()

    if wordlist_file.endswith('.npz'): 
        dataset = AutoencoderDataset(wordlist_file)
        os.symlink(wordlist_file, dataset.file_to_write())
    else: 
        words = open(wordlist_file).readlines()
        dataset = AutoencoderDataset(initial_size=len(words))
        for w in words: 
            for wi in create_word_images(w, **kwargs): 
                comp_arr = aui.decompose_and_resize(wi, size, size)
                comp_arr = remove_zeros_and_reshape(comp_arr)
                dataset.add_components(comp_arr)

        dataset.write(save_dir() + "autoencoder-dataset.npz")
    
    autoencoder, decoder, encoder = train_model(dataset,
                                                autoencoder, 
                                                decoder, 
                                                encoder)
    
    reconstruction_test(dataset, autoencoder)
    
    return (autoencoder, decoder, encoder)

def reconstruction_test(dataset, autoencoder):

    x_train = dataset.as_dataset()
    selection = np.random.randint(0, x_train.shape[0], 1000)
    x_test = x_train[selection]
    decoded_imgs = autoencoder.predict(x_test)
    LOG.debug("=== x_test.shape ===")
    LOG.debug(x_test.shape)
    LOG.debug("=== decoded_imgs.shape ===")
    LOG.debug(decoded_imgs.shape)
    

    for i, orig in enumerate(x_test): 
        oc, ox, oy = orig.shape
        out_img = np.zeros(shape=(oc, ox, oy*2), dtype=np.uint8)
        out_img[:, :, :oy] = np.floor(orig * 255)
        decoded = decoded_imgs[i]
        out_img[:, :, oy:] = np.floor(decoded * 255)
        out_img = out_img.swapaxes(0, 1).swapaxes(1, 2)
        out_file = "{}/decoded-{}.png".format(save_dir(), i)
        LOG.debug("=== out_img.shape ===")
        LOG.debug(out_img.shape)
        
        cv2.imwrite(out_file, out_img)

def main(): 
    parser = argparse.ArgumentParser(
        description="""
        Builds incremental using given word list
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('word-list',
                        help="Loads the letter group dataset from npz file")
    args = vars(parser.parse_args())

    text_detection_ac_v1(args['word-list'])

if __name__ == '__main__': 
    main()
