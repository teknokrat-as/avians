
# import keras.models as km
# import keras.layers.core as klc
# import keras.layers.embeddings as kle
# import keras.layers.recurrent as klr
# import keras.callbacks as kc

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, Callback

import generate_dataset as gd

from datetime import datetime as dt
import cv2
import numpy as np
import argparse
import os
import itertools as it
import avians.util.image as aui
from avians.util.decorators import static_vars
import socket

import multiprocessing as mp

import gc
import boto3
from slacker import Slacker

import logging
LOG = logging.getLogger(__name__)

def process_image(ti): 
    print("Processing: {}".format(ti))
    img = cv2.imread(ti, 0)
    LOG.debug("=== ti ===")
    LOG.debug(ti)
    
    contours = u.contours(img)
    # LOG.debug("=== contours ===")
    # LOG.debug(contours)
    
    # each element is a tuple (cnt, holes, level)
    features = np.zeros(shape=(len(contours), ), dtype=u.CONTOUR_FEATURE_COLUMNS)
    for i, contour in enumerate(contours): 
        cnt, holes, level = contour
        u.contour_features(cnt, holes, level, img, features[i])
#        features[i]["class"] = class_value
    return features

def process_image_for_CNN(ti, rows=32, cols=32, reduced_colors=10,
                          feature_func=gd.resized_contour_region): 
    img = cv2.imread(ti)
    print("Processing: {} ({}x{})".format(ti, 
                                          img.shape[0], 
                                          img.shape[1]))
    contours = aui.contours_via_color_quantization(img)
    # Filter very small elements
    threshold_size = img.shape[0] * img.shape[1] * 0.0001
    contours = aui.filter_small_contours(contours, threshold_size)
    
    # print("Contour & Holes: {}".format(len(contour_and_holes)))
    # LOG.debug("=== large.shape ===")
    # LOG.debug(large.shape)
    # LOG.debug("=== contours ===")
    # LOG.debug(contours)
    
    # each element is a tuple (cnt, holes, level)
    ch = 1 if img.shape == 2 else 3
    features = np.zeros(shape=(len(contours), 
                               ch, rows, cols), dtype=np.uint8)
    for i, contour in enumerate(contours):
        feature = feature_func(img, contour, rows, cols)
        if len(feature.shape) == 3: 
            # Convert rowsXcolsX3 image to 3XrowsXcols
            features[i] = feature.swapaxes(0, 2).swapaxes(1, 2)
        else: 
            features[i] = feature

    return features


def process_image_for_CNN_multichannel(ti, channels=3, rows=32, cols=32): 
    img = cv2.imread(ti)
    print("Processing: {} ({}x{}x{})".format(ti, img.shape[0], img.shape[1], img.shape[2]))
    if channels == 1: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lenr = np.ceil(img.shape[0]/rows)
    lenc = np.ceil(img.shape[1]/cols)
    features = np.zeros(shape=(lenr * lenc, rows, cols, channels), dtype=np.uint8)
    for (i, j) in np.ndindex(lenr, lenc): 
        s = img[i*rows:(i+1)*rows, j*cols:(j+1)*cols]
        sr = s.shape[0]
        sc = s.shape[1]
        features[i * lenc + j, :sr, :sc, :] = s

    # feature_array = np.array([img[i*rows:(i+1)*rows, j*cols:(j+1)*cols, :] for (i, j) in np.ndindex(lenr,lenc)])
    swapped = features.swapaxes(1, 3).swapaxes(2, 3)
    return swapped

def process_image_for_CNN_multichannel_mp(args):
    return process_image_for_CNN_multichannel(*args)

def process_image_for_CNN_mp(kwargs): 
    return process_image_for_CNN(**kwargs)

def generate_words_and_process_image_mp(kwargs): 
    with open(kwargs["wordlist"]) as wordlist_f: 
        wordlist = wordlist_f.readlines()

    f = gd.generate_random_arabic_text_image(wordlist)
    return process_image_for_CNN(f, 
                                 rows=kwargs["rows"], 
                                 cols=kwargs["cols"], 
                                 feature_func=kwargs["feature_func"])

def generate_dataset_by_words(positive_wordlist, non_text_images,
                              kernel_x=32, kernel_y=32, 
                              positive_reduced_colors=2,
                              negative_reduced_colors=10,
                              use_mp=False):
    print("Generating Negative Set")
    pool = mp.Pool(max(mp.cpu_count() - 1, 1))
    negative_args = [{"ti": f, 
                      "rows": kernel_x, 
                      "cols": kernel_y,
                      "reduced_colors": negative_reduced_colors,
                      "feature_func": gd.resized_contour_region}  
                     for f in non_text_images]
    if use_mp:
        negative_results = pool.map(process_image_for_CNN_mp, negative_args)
    else: 
        negative_results = [process_image_for_CNN_mp(a) for a in negative_args]
    negative_items = sum([len(nr) for nr in negative_results]) 

    # Produce negative_items * 2 positive_items with words
    positive_items_target = 2 * negative_items
    positive_results = []
    positive_items = 0
    batch_size = 30
    while positive_items < positive_items_target: 
        # Produce batch size images
        batch_args = [{"wordlist": positive_wordlist,
                       "rows": kernel_x, 
                       "cols": kernel_y,
                       "reduced_colors": positive_reduced_colors,
                       "feature_func": gd.resized_contour_region} 
                      for i in range(batch_size)]
        if use_mp: 
            batch_results = pool.map(generate_words_and_process_image_mp, 
                                     batch_args)
        else: 
            batch_results = [generate_words_and_process_image_mp(a) 
                             for a in batch_args]

        batch_items = sum([len(br) for br in batch_results])
        positive_results += batch_results
        positive_items += batch_items


    n_channels = 3
    return combine_features(positive_results, negative_results, 
                            n_channels, kernel_x, kernel_y)

def combine_features(positive_elements, negative_elements, channels, kernel_x, kernel_y):
    positive_items = sum([len(pr) for pr in positive_elements])
    negative_items = sum([len(nr) for nr in negative_elements]) 
    print("Positive Items: {}".format(positive_items))
    print("Negative Items: {}".format(negative_items))

    total_size = positive_items + negative_items

    backup_file_name = "dataset-positive-negative-elements-backup-{}.npz".format(dt.now().strftime("%F-%T"))
    np.savez_compressed(backup_file_name, positive_elements=positive_elements, negative_elements=negative_elements)
    save_to_s3(backup_file_name)

    gc.collect()

    # combine features

    # combined_features = np.zeros(shape=(total_size,), dtype=u.CONTOUR_FEATURE_COLUMNS)
    x_train = np.zeros(shape=(total_size, channels, kernel_x, kernel_y), dtype=np.uint8)
    LOG.debug("=== x_train.shape ===")
    LOG.debug(x_train.shape)

    y_train = np.zeros(shape=(total_size, ), dtype=np.uint8)
    LOG.debug("=== y_train.shape ===")
    LOG.debug(y_train.shape)

    cursor = 0
    for feature_arr in positive_elements:
        step, c, w, h = feature_arr.shape
        LOG.debug("=== step ===")
        LOG.debug(step)
        
        LOG.debug("=== feature_arr.shape ===")
        LOG.debug(feature_arr.shape)
        LOG.debug("=== x_train[cursor:(cursor+step)].shape ===")
        LOG.debug(x_train[cursor:(cursor+step)].shape)
        x_train[cursor:cursor+step] = feature_arr
        y_train[cursor:cursor+step] = 1
        cursor += step

    for feature_arr in negative_elements: 
        step, c, w, h = feature_arr.shape
        x_train[cursor:cursor+step] = feature_arr
        y_train[cursor:cursor+step] = 0
        cursor += step

    return x_train, y_train


def generate_dataset(text_imgs, non_text_imgs): 
    channels = 3
    kernel_x = 32
    kernel_y = 32
    pool = mp.Pool(max(mp.cpu_count() - 1, 1))
    positive_args = [(f, channels, kernel_x, kernel_y) for f in text_imgs]
    negative_args = [(f, channels, kernel_x, kernel_y) for f in non_text_imgs]
    positive_results = pool.map(process_image_for_CNN_mp, positive_args)
    negative_results = pool.map(process_image_for_CNN_mp, negative_args)
    # positive_results = [process_image_for_CNN_mp(f) for f in positive_args]
    # negative_results = [process_image_for_CNN_mp(f) for f in negative_args]
    
    print("Text Images: {}".format(len(positive_results)))
    print("Non-Text Images: {}".format(len(negative_results)))

    positive_items = sum([len(pr) for pr in positive_results])
    negative_items = sum([len(nr) for nr in negative_results]) 

    print("Positive Items: {}".format(positive_items))
    print("Negative Items: {}".format(negative_items))

    return combine_features(positive_results, negative_results, channels, kernel_x, kernel_y)

# FILTER_COLUMNS = ["x", "y", "width", "height", "area", "class"]
# CLASSIFICATION_FEATURES=[f[0] for f in u.CONTOUR_FEATURE_COLUMNS if f[0] not in FILTER_COLUMNS]
CLASSIFICATION_FEATURES = [
    "relative_width",
    "relative_height",
    "relative_area",
    "hull_area",
    "relative_hull_area",
    "rect_area",
    "relative_rect_area",
    "aspect_ratio",
    "extent",
    # "solidity",
    "convexity_defect_dist",
    # "normalized_convexity_defect_distance",
    # "major_axis_length",
    # "minor_axis_length",
    # "orientation_angle",
    "color_mean",
    "color_std",
    # "distance_transform_mean",
    # "distance_transform_std",
    "distance_transform_max",
    "num_holes",
    "hierarchy_level"]

CLASSES = [0, 1]


def create_model(whole_dataset, feature_columns): 
    
    positive_n = ((whole_dataset["class"] == 1).nonzero())[0].size
    negative_n = ((whole_dataset["class"] == 0).nonzero())[0].size

    samples_from_each = min(positive_n, negative_n)
    positive_data = np.random.permutation(whole_dataset[whole_dataset["class"] == 1])[:samples_from_each]
    negative_data = np.random.permutation(whole_dataset[whole_dataset["class"] == 0])[:samples_from_each]
    dataset = np.random.permutation(np.concatenate((positive_data, negative_data)))
    selected = dataset[feature_columns]
    the_data = selected.view(np.float64).reshape(selected.shape + (-1, ))

    print("the_data.shape: {}".format(the_data.shape))
    labels = dataset["class"].view(np.int).reshape(dataset["class"].shape + (-1,))
    
    print("Creating model")
    model = Sequential()

    # Below MLP gives around 0.95 val_acc maximum. 
    # model.add(klc.Dense(256, input_shape=(len(feature_columns),), init='uniform', activation='relu'))
    # model.add(klc.Dropout(0.4))
    # model.add(klc.Dense(256, activation='relu'))
    # model.add(klc.Dropout(0.2))
    # model.add(klc.Dense(256, activation='relu'))
    # model.add(klc.Dropout(0.1))
    # model.add(klc.Dense(output_dim=1, activation='sigmoid'))

    max_len, max_features = the_data.shape
#    model.add(Embedding(max_features, 128, input_length=max_len, dropout=0.5))
# model.add(Dense(128, input_shape=(len(feature_columns),), init='uniform', activation='relu'))
    model.add(LSTM(128, input_shape=the_data.shape, dropout_W=0.5, dropout_U=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    print("Compiling Model: {}".format(model))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  class_mode='binary',
                  metrics=["accuracy"])

    stop_early = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    print("Fitting Model: {}".format(model))
    print("Data: {}".format(the_data))
    print("Labels: {}".format(labels))
    model.fit(the_data, 
              labels, 
              nb_epoch=100, 
              batch_size=32, 
              validation_split=0.3, 
              verbose=1, 
              callbacks=[stop_early])

    return model


def build_cnn_model(num_channels, img_rows, img_cols, nb_filters=32, nb_conv=3, nb_pool=2):

    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(num_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', 
                  optimizer='adadelta', 
                  metrics=["accuracy"])
    return model

def create_cnn(x_train, 
               y_train, 
               nb_filters=32, 
               nb_conv=3, 
               nb_pool=2, 
               batch_size=64, 
               nb_epoch=100): 

    num_elements, num_channels, img_rows, img_cols = x_train.shape

    model = build_cnn_model(num_channels, img_rows, img_cols, nb_filters, nb_conv, nb_pool)

    stop_early = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    slacker_progress = SlackerProgress()
    s3_uploader = S3Uploader()

    print("Fitting Model: {}".format(model))
    print("Data Shape: {}".format(x_train.shape))
    print("Class Shape: {}".format(y_train.shape))

    model.fit(x_train, 
              y_train, 
              batch_size=batch_size, 
              nb_epoch=nb_epoch,
              verbose=1, validation_split=0.1, callbacks=[stop_early, slacker_progress, s3_uploader])
    return model

@static_vars(slack=Slacker("xoxp-3061246616-3061246618-19515477046-aab87bd1f0"))
def slacker_message(msg):
    slacker_message.slack.chat.post_message("#output", msg)

class SlackerProgress(Callback): 
    def on_train_begin(self, logs={}): 
        hostname = socket.gethostname()
        samples = self.params["nb_sample"]
        epoch = self.params["nb_epoch"]
        batch_size = self.params["batch_size"]
        msg = """Beginning training on {}
Samples: {}
Epoch: {}
Batch Size: {}
        """.format(hostname, samples, epoch, batch_size)
        slacker_message(msg)
    
    def on_epoch_end(self, epoch, logs={}): 
        hostname = socket.gethostname()
        msg = """End of Epoch {} in {}

*loss:*  {:.2f}
*acc:* {:.2f}
*val_loss:* {:.2f}
*val_acc:* {:.2f}

""".format(epoch, hostname, logs["loss"], logs["acc"], logs["val_loss"], logs["val_acc"])
        slacker_message(msg)


def save_to_s3(filename): 
    s3 = boto3.resource("s3", region_name="us-east-1")
    f = open(filename, "rb")
    s3.Bucket("avians-data").put_object(Key=os.path.basename(filename), Body=f)
    f.close()
    slacker_message("Uploaded Dataset to S3: avians-data/{}".format(filename))

class S3Uploader(Callback):

    def __init__(self): 
        super().__init__()
        self.s3 = boto3.resource("s3", region_name="us-east-1")
        self.modelkey = dt.now().strftime("%F-%T")

    def on_train_begin(self, logs={}): 
        key = "text-detection-model--{}.json".format(self.modelkey)
        filename = "/tmp/" + key
        ff = open(filename, "w")
        ff.write(self.model.to_json())
        ff.close()
        data = open(filename, "rb")
        self.s3.Bucket("avians-data").put_object(Key=key, Body=data)
        slacker_message("Uploaded to S3: avians-data/{}".format(key))

    def on_epoch_end(self, epoch, logs={}): 
        key = "text-detection-model--{}-epoch-{}--val_acc-{:.2f}--val_loss-{:.2f}.h5".format(
            self.modelkey, epoch, logs["val_acc"], logs["val_loss"])
        filename = "/tmp/" + key
        self.model.save_weights(filename)
        data = open(filename, "rb")
        self.s3.Bucket("avians-data").put_object(Key=key, Body=data)
        slacker_message("Uploaded to S3: avians-data/{}".format(key))

def train_cnn_from_dataset(load_from, output_model): 
    print("Loading Dataset From: {}".format(load_from))
    f_in = np.load(load_from)
    x_train = f_in["x_train"]
    y_train = f_in["y_train"]

    rng_state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)

    model = create_cnn(x_train, y_train)
    json_model = model.to_json()
    open(output_model + '.json', 'w').write(json_model)
    model.save_weights(output_model + ".h5")
    return model

def train_cnn_by_images(positive_set, negative_set, output_model):
    if os.path.isdir(positive_set): 
        positive_samples = [os.path.join(positive_set, f) for f in os.listdir(positive_set)]
    else: 
        positive_samples = [positive_set]

    if os.path.isdir(negative_set): 
        negative_samples = [os.path.join(negative_set, f) for f in os.listdir(negative_set)]
    else: 
        negative_samples = [negative_set]

    print("Positive Files: {}".format(positive_samples))
    print("Negative Files: {}".format(negative_samples))

    x_train, y_train = generate_dataset(positive_samples, negative_samples)
    dataset_name = "text-detection-by-images-dataset--shape-{}--{}.npz".format(str(x_train.shape), dt.strftime(dt.now(), "%F-%T"))
    np.savez_compressed(dataset_name, x_train=x_train, y_train=y_train)

    save_to_s3(dataset_name)
    return train_cnn_from_dataset(dataset_name, output_model)

def train_cnn_by_words(positive_wordlist, negative_set, output_model, use_mp=False): 
    # words = open(positive_wordlist).readlines()

    if os.path.isdir(negative_set): 
        negative_samples = [os.path.join(negative_set, f) for f in os.listdir(negative_set)]
    else: 
        negative_samples = [negative_set]

    x_train, y_train = generate_dataset_by_words(positive_wordlist, negative_samples, use_mp=use_mp)
    dataset_name = "text-detection-by-words-dataset--shape-{}--{}.npz".format(str(x_train.shape), dt.strftime(dt.now(), "%F-%T"))
    np.savez_compressed(dataset_name, x_train=x_train, y_train=y_train)
    save_to_s3(dataset_name)

    return train_cnn_from_dataset(dataset_name, output_model)

def load_cnn_model(model_name): 
    model = model_from_json(open(model_name + '.json').read())
    model.compile(loss='binary_crossentropy', 
                  optimizer='adadelta', 
                  metrics=["accuracy"])
    model.load_weights(model_name + ".h5")
    return model
    
def train_rnn(positive_set, negative_set, output_model, load_from=None): 
    if load_from is None: 
        if os.path.isdir(positive_set): 
            positive_samples = [os.path.join(positive_set, f) for f in os.listdir(positive_set)]
        else: 
            positive_samples = [positive_set]

        if os.path.isdir(negative_set): 
            negative_samples = [os.path.join(negative_set, f) for f in os.listdir(negative_set)]
        else: 
            negative_samples = [negative_set]

        print("Positive Files: {}".format(positive_samples))
        print("Negative Files: {}".format(negative_samples))
        dataset = generate_dataset(positive_samples, negative_samples)
        dataset_filename = "{}-{}-{}".format(positive_set.replace("/", "~"), negative_set.replace("/", "-"), dt.strftime(dt.now(), "%F-%T"))
        np.savez_compressed(dataset_filename, dataset=dataset)
        print("Dataset Saved: {}".format(dataset_filename))
    else: 
        print("Loading Dataset From: {}".format(load_from))
        f_in = np.load(load_from)
        dataset = f_in["dataset"]

    for feature in it.combinations(CLASSIFICATION_FEATURES, 14):
        featuqre = list(feature)
        print("Creating Model for {}".format(feature))
        model = create_model(dataset, feature)
        model.save_weights(output_model + "-".join(feature) + ".h5")


def build_mlp_model(input_dim): 

    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # "class_mode" defaults to "categorical". For correctly displaying accuracy
    # in a binary classification problem, it should be set to "binary".

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=["accuracy"])

    
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
    parser.add_argument('--positive-set', 
                        help="Directory for text images", 
                        default=None)
    parser.add_argument('--positive-wordlist', 
                        help="File to generate text images", 
                        default=None)
    parser.add_argument('--negative-set', 
                        help="Directory for non text images", 
                        default=None)
    parser.add_argument('--no-multiprocessing', 
                        help="Don't use multiprocessing module",
                        action='store_true',
                        default=False)

    args = vars(parser.parse_args())
    pos_s = args["positive_set"]
    pos_wl = args["positive_wordlist"]
    neg_s = args["negative_set"]
    out_m = args["output_model"]
    loa_f = args["load_from"]
    use_mp = not args["no_multiprocessing"]

    print(args)
    if loa_f is not None: 
        train_cnn_from_dataset(loa_f, out_m)
    elif pos_s is None and neg_s is not None and pos_wl is not None: 
        train_cnn_by_words(pos_wl, neg_s, out_m, use_mp=use_mp)
    elif pos_s is not None and neg_s is not None: 
        train_cnn_by_images(pos_s, neg_s, out_m)
    else: 
        print("""You need to supply, either
 (a) --load-from [a previously created dataset]
 (b) --positive-set [image file or dir] --negative-set [image file or dir]
 (c) --positive-wordlist [words file] --negative-set [image file or dir]
        """)


if __name__ == "__main__": 
    main()
