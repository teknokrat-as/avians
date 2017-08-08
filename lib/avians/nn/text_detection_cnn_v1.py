
import os
import numpy as np
import numpy.random as nr
from datetime import datetime as dt
import argparse
import gc

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping

import avians.util.image as aui
import avians.nn.common as anc
import avians.util.arabic as aua
from avians.util.decorators import static_vars

import logging
LOG = logging.getLogger(__name__)

NEGATIVE_IMAGE_RESOLUTIONS = [(1280, 720),
                              # (854, 480),
                              # (640, 360), 
                              (480, 320)]
CNN_INPUT_SIZE = 32
IMAGE_STRIDES = 16

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

def train_cnn_by_words(wordlist_file,
                       negative_dir, 
                       output_model, 
                       use_mp=False): 

    print("Preparing Negative Set")
    negative_set = prepare_negative_set(negative_dir)
    n_neg = negative_set.shape[0]
    # Produce twice as much positive images
    print("Preparing Positive Set")
    positive_set = prepare_positive_set(wordlist_file, n_neg * 2)
    dataset_name = save_dir() + "dataset-by-words-negative-{}-positive-{}-{}.npz".format(str(negative_set.shape[0]), 
                                                                                         str(positive_set.shape[0]),
                                                                                         dt.strftime(dt.now(), "%F-%T"))
    print("Saving Dataset to {}".format(dataset_name))
    np.savez_compressed(dataset_name, 
                        positive_set=positive_set, 
                        negative_set=negative_set)
    return train_cnn_from_dataset(dataset_name, output_model)


def train_cnn_from_dataset(load_from, output_model): 
    print("Loading Dataset From: {}".format(load_from))
    ds_in = np.load(load_from)
    positive_set = ds_in["positive_set"]
    negative_set = ds_in["negative_set"]
    positive_set = positive_set.swapaxes(1, 3).swapaxes(2, 3)
    negative_set = negative_set.swapaxes(1, 3).swapaxes(2, 3)
    positive_y = np.ones(shape=(positive_set.shape[0], 1))
    negative_y = np.ones(shape=(negative_set.shape[0], 1))
    LOG.debug("=== positive_y.shape ===")
    LOG.debug(positive_y.shape)
    LOG.debug("=== negative_y.shape ===")
    LOG.debug(negative_y.shape)
    y_train = np.vstack((np.ones(shape=(positive_set.shape[0], 1)),
                         np.zeros(shape=(negative_set.shape[0], 1))))
    x_train_shape=(positive_set.shape[0] + negative_set.shape[0],
                  positive_set.shape[1],
                  positive_set.shape[2],
                  positive_set.shape[3])
    print("Merging Positive and Negative Sets into Memmap File: {}".format(x_train_shape))
    x_train = np.memmap("x_train.memmap", 
                        dtype=positive_set.dtype,
                        shape=x_train_shape, mode="w+")
    x_train[:positive_set.shape[0]] = positive_set
    x_train[positive_set.shape[0]:] = negative_set
    # move channel axis to 2
    del positive_set
    del negative_set
    gc.collect()
    LOG.debug("=== x_train.shape ===")
    LOG.debug(x_train.shape)
    anc.shuffle_parallel(x_train, y_train)
    model = create_cnn(x_train, y_train)
    anc.save_and_upload_model(model, save_dir())
    return model

def generate_random_arabic_text_images(words, 
                                       fonts,
                                       n_images = 32,
                                       num_words_min = 1,
                                       num_words_max = 10,
                                       dpi_min=50, 
                                       dpi_max=600,
                                       bg_min = 0,
                                       bg_max = 255,
                                       fg_min = 0,
                                       fg_max = 255):
    def random_img(): 
        num_words_in_text = nr.randint(num_words_min, num_words_max)
        text = " ".join([w.strip() for w in list(nr.choice(words, num_words_in_text))])
        fg = nr.randint(fg_min, fg_max)
        bg = nr.randint(bg_min, bg_max)
        font = nr.choice(fonts)
        dpi = nr.randint(dpi_min, dpi_max)
        return aui.get_gray_word_image(text, font, dpi, bg, fg)

    return [random_img() for i in range(n_images)]

def prepare_negative_set(negative_dir): 

    # words = open(positive_wordlist).readlines()
    print("Listing Image Files")
    negative_files = aui.file_list(negative_dir)
    print("Loading Image Files")
    negative_images = [img for img, p in aui.image_file_list(negative_files)]
    # Get TV / video resolutions
    print("Resizing Images to Video Resolutions")
    resized_images = negative_images + aui.resize_image_list(negative_images,
                                                             NEGATIVE_IMAGE_RESOLUTIONS)
    print("A total of {} images done".format(len(resized_images)))
    negative_patches = aui.split_image_list(resized_images, 
                                            CNN_INPUT_SIZE,
                                            CNN_INPUT_SIZE,
                                            IMAGE_STRIDES,
                                            IMAGE_STRIDES)
    
    print("A total of {} patches collected".format(negative_patches.shape[0]))
    # Normalize to keep around 0
    negative_set = (negative_patches - 128) / 128
    return negative_set

def prepare_positive_set(wordlist_file,
                         n_elements): 
    fonts = aua.system_arabic_fonts()
    print("Listed {} Arabic Fonts from System".format(len(fonts)))
    with open(wordlist_file) as f:
        words = [w.strip() for w in f.readlines()]
    
    total_n = 0
    array_list = []
    print("Generating Word Images")
    while total_n <= n_elements:
        images = generate_random_arabic_text_images(words, fonts)
        image_patches = aui.split_image_list(images, 
                                             CNN_INPUT_SIZE,
                                             CNN_INPUT_SIZE,
                                             IMAGE_STRIDES,
                                             IMAGE_STRIDES)
        # remove patches with a uniform value
        image_patches = image_patches[np.nonzero(
            np.apply_over_axes(
                np.var, image_patches, [1, 2]))[0]]   

        array_list.append(image_patches)
        total_n += image_patches.shape[0]
        print(total_n, end=",", flush=True)

    res_array = (np.vstack(array_list) - 128) / 128
    return res_array

def create_cnn(x_train, 
               y_train, 
               nb_filters=32, 
               nb_conv=3, 
               nb_pool=2, 
               batch_size=64,
               nb_epoch=100): 

    num_elements, num_channels, img_rows, img_cols = x_train.shape

    model = build_cnn_model(num_channels, img_rows, img_cols, nb_filters, nb_conv, nb_pool)

    stop_early = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

    print("Fitting Model: {}".format(model))
    print("Data Shape: {}".format(x_train.shape))
    print("Class Shape: {}".format(y_train.shape))

    model.fit(x_train, 
              y_train, 
              batch_size=batch_size, 
              nb_epoch=nb_epoch,
              verbose=1, validation_split=0.1, callbacks=[stop_early])
    return model



def build_cnn_model(num_channels, 
                    img_rows, 
                    img_cols, 
                    nb_filters=32, 
                    nb_conv=3, 
                    nb_pool=2):

    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(num_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, 
                            nb_conv, 
                            nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.10))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(num_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, 
                            nb_conv, 
                            nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.10))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', 
                  optimizer='adadelta', 
                  metrics=["accuracy"])
    return model


def main():

    parser = argparse.ArgumentParser(description="""
    Builds a keras model using the text and non text images. 
    
    """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-model',
                        help="Keras model to output",
                        default="text-detection-model-{}".format(dt.now().strftime("%F-%T")))
    parser.add_argument('--load-from',
                        help="Loads the dataset from saved npy file",
                        default=None)
    # parser.add_argument('--positive-set',
    #                     help="Directory for text images",
    #                     default=None)
    parser.add_argument('--positive-wordlist',
                        help="File to generate text images",
                        default=None)
    parser.add_argument('--negative-set',
                        help="Directory for non text images",
                        default=None)
    # parser.add_argument('--no-multiprocessing',
    #                     help="Don't use multiprocessing module",
    #                     action='store_true',
    #                     default=False)

    args = vars(parser.parse_args())
#    pos_s = args["positive_set"]
    pos_wl = args["positive_wordlist"]
    neg_s = args["negative_set"]
    out_m = args["output_model"]
    loa_f = args["load_from"]
#    use_mp = not args["no_multiprocessing"]

    print(args)
    if loa_f is not None: 
        train_cnn_from_dataset(loa_f, out_m)
    elif neg_s is not None and pos_wl is not None: 
        train_cnn_by_words(pos_wl, neg_s, out_m)
    # elif pos_s is not None and neg_s is not None: 
    #     train_cnn_by_images(pos_s, neg_s, out_m)
 #    else: 
 #        print("""You need to supply, either
 # (a) --load-from [a previously created dataset]
 # (b) --positive-set [image file or dir] --negative-set [image file or dir]
 # (c) --positive-wordlist [words file] --negative-set [image file or dir]
 #        """)


if __name__ == "__main__": 
    main()
