
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Convolution2D, Flatten, Reshape, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.utils import np_utils
import os
import re

from datetime import datetime as dt
import cv2
import numpy as np
import avians.util.image as aui

# from typing import List, Tuple, Optional, TypeVar

import boto3
from slacker import Slacker
from avians.util.decorators import static_vars
import socket
import logging
LOG = logging.getLogger(__name__)


LABEL_EXTRACTION_REGEX_LIST = [
    (r"(.*)/cc-(.+)-size-[0-9]+-[0-9]+\.png", "\g<2>"),
    (r"(.*)/cc-(.+)-index-[0-9]+.*\.png", "\g<2>"), 
    (r"(.*)/cc-(.+)-(.+)-loc-[0-9-]+.*\.png", "\g<2>")] 

# SOME AWS and SLACKER UTILS

@static_vars(slack=Slacker("xoxp-3061246616-3061246618-19515477046-aab87bd1f0"))
def slacker_message(msg):
    try: 
        slacker_message.slack.chat.post_message("#output", msg)
    except: 
        print("Cannot send message to slack: {}".format(msg))

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

""".format(epoch, hostname, 
           logs["loss"], logs["acc"], logs["val_loss"], logs["val_acc"])
        slacker_message(msg)

def json_and_h5_filenames(fn): 
    if fn.endswith(".json"): 
        base = fn[:-5]
    elif fn.endswith(".h5"): 
        base = fn[:-3]
    else: 
        base = fn
        
    return (base + ".json", base + ".h5")

def load_models(**kwargs):
    """Loads multiple keras models from h5 and json files. 

    :param kwargs: A keyword argument list of models
    
    :returns: A dictionary where keys are argument names and values are keras models
    """

    
    models = {}
    for modelname, filename in kwargs.items(): 
        json_file, weight_file = json_and_h5_filenames(filename)

        with open(json_file) as jf: 
            json_string = jf.read()
        
        model = model_from_json(json_string)
        model.compile(loss='categorical_crossentropy', 
                      optimizer='rmsprop', 
                      metrics=["accuracy"])
        model.load_weights(weight_file)
        models[modelname] = model
    return models

def load_datasets(**dataset_files):
    """Loads numpy datasets from multiple npz files. Each file contains multiple numpy data
    
    :param dataset_files: keyword argument list of dataset names and files
    :returns: A dictionary where keys are argument names and values are dictionaries. Each value dictionary corresponds to the dataset file"""

    datasets = {}

    for k, fname in dataset_files.items(): 
        datasets[k] = {}
        ds = np.load(fname)

        for dn in ds.keys():
            datasets[k][dn] = ds[dn]
    return datasets


def save_and_upload_model(model, save_dir='/tmp/', model_key="", upload=False):
    model_file = 'model-{}-{}'.format(model_key, dt.strftime(dt.now(), '%F-%T'))
    json_model = model.to_json()
    json_filename = model_file + ".json"
    open(save_dir + json_filename, 'w').write(json_model)
    weights_file = save_dir + model_file + ".h5"
    model.save_weights(weights_file)
    if upload: 
        save_to_s3(json_filename)
        save_to_s3(weights_file)


def save_to_s3(filename): 
    s3 = boto3.resource("s3", region_name="us-east-1")
    f = open(filename, "rb")
    s3.Bucket("dervaze-models").put_object(
        Key=os.path.basename(filename), Body=f)
    f.close()
    slacker_message(
        "Uploaded to S3: dervaze-models/{}".format(filename))

class S3Uploader(Callback):
    def __init__(self, modelkey=None): 
        super().__init__()
        self.s3 = boto3.resource("s3", region_name="us-east-1")
        if modelkey is None: 
            self.modelkey = dt.now().strftime("%F-%T")
        else:
            self.modelkey = modelkey

    def on_train_begin(self, logs={}): 
        key = "harfbin-model--{}.json".format(self.modelkey)
        filename = "/tmp/" + key
        ff = open(filename, "w")
        ff.write(self.model.to_json())
        ff.close()
        data = open(filename, "rb")
        self.s3.Bucket("dervaze-models").put_object(Key=key, Body=data)
        slacker_message("Uploaded to S3: dervaze-models/{}".format(key))

    def on_epoch_end(self, epoch, logs={}): 
        key = "harfbin-model--{}-epoch-{}--val_acc-{:.2f}--val_loss-{:.2f}.h5".format(  # noqa
            self.modelkey, epoch, logs["val_acc"], logs["val_loss"])
        filename = "/tmp/" + key
        self.model.save_weights(filename)
        data = open(filename, "rb")
        self.s3.Bucket("dervaze-models").put_object(Key=key, Body=data)
        slacker_message("Uploaded to S3: dervaze-models/{}".format(key))

def model_filenames(fn): 
    if fn.endswith(".json"): 
        base = fn[:-5]
    elif fn.endswith(".h5"): 
        base = fn[:-3]
    else: 
        base = fn
    return (base + ".json", base + ".h5")

def load_keras_model(filename, 
                     loss='categorical_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy']): 

    json_file, weight_file = model_filenames(filename)
    with open(json_file) as jf: 
        json_string = jf.read()

    model = model_from_json(json_string)
    model.compile(loss=loss, 
                  optimizer=optimizer, 
                  metrics=metrics)
    model.load_weights(weight_file)
    return model

def get_image_label_list(dataset_dir, invert_images=False):
    image_label_list = []
    if dataset_dir:
        image_files = aui.file_list(os.path.expandvars(dataset_dir))
        LOG.debug("=== len(image_files) ===")
        LOG.debug(len(image_files))
        the_pat = ""
        the_subst = ""
        for pat, subst in LABEL_EXTRACTION_REGEX_LIST: 
            if re.match(pat, image_files[0]):
                the_pat = pat
                the_subst = subst

        if the_pat == "": 
            raise Exception("Cannot extract label from filename: {}".format(
                image_files[0]))

        for f in image_files:
            label = re.sub(the_pat, the_subst, f)
            if label is not None: 
                img = cv2.imread(f, 0)
                if invert_images: 
                    img = aui.invert_img(img)
                image_label_list.append((img, label))

    return image_label_list

def preprocess_image(img, 
                     invert=False, 
                     sizex=64, 
                     sizey=64, 
                     dist_transform=True): 
    if invert: 
        img = aui.invert_img(img)

    if sizex > 0 and sizey > 0: 
        img = cv2.resize(img, (sizex, sizey), interpolation=cv2.INTER_AREA)
    elif sizex > 0: 
        sizey = img.shape[1]
        img = cv2.resize(img, (sizex, sizey))
    elif sizey > 0: 
        sizex = img.shape[0]
        img = cv2.resize(img, (sizex, sizey))

    if dist_transform:
        img = distance_transform(img)

    return img


def prepare_validation_data(x_train, y_train): 
    rng_state = np.random.get_state()
    x_val = x_train.copy()
    np.random.shuffle(x_val)
    np.random.set_state(rng_state)
    y_val = y_train.copy()
    np.random.shuffle(y_val)

    if x_val.shape[0] > 10000: 
        x_val = x_val[:10000]
        y_val = y_val[:10000]

    return (x_val, y_val)

def distort_image(img):
    
    distortions = []

    distortions.append(img)
    # distortions.append(cv2.resize(img, 
    #                               dsize=(0, 0), fx=1.2, fy=1, 
    #                               interpolation=cv2.INTER_NEAREST))
    # distortions.append(cv2.resize(img, 
    #                               dsize=(0, 0), fx=1.5, fy=1, 
    #                               interpolation=cv2.INTER_NEAREST))
    distortions.append(cv2.resize(img, 
                                  dsize=(0, 0), fx=2, fy=1, 
                                  interpolation=cv2.INTER_AREA))
    # distortions.append(cv2.resize(img, 
    #                               dsize=(0, 0), fx=3, fy=1, 
    #                               interpolation=cv2.INTER_NEAREST))
    # distortions.append(cv2.resize(img, 
    #                               dsize=(0, 0), fx=1, fy=1.2, 
    #                               interpolation=cv2.INTER_NEAREST))
    # distortions.append(cv2.resize(img, 
    #                               dsize=(0, 0), fx=1, fy=1.5, 
    #                               interpolation=cv2.INTER_NEAREST))
    distortions.append(cv2.resize(img, 
                                  dsize=(0, 0), fx=1, fy=2, 
                                  interpolation=cv2.INTER_AREA))
    # distortions.append(cv2.resize(img, 
    #                               dsize=(0, 0), fx=1, fy=3, 
    #                               interpolation=cv2.INTER_NEAREST))

    ellipse53 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    # ellipse55 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ellipse33 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    erode33 = cv2.erode(img, ellipse33)
    # erode53 = cv2.erode(img, ellipse53)
    # erode55 = cv2.erode(img, ellipse55)
    # erode55x2 = cv2.erode(erode55, ellipse55)
    # erode55x3 = cv2.erode(erode55x2, ellipse55)

    dilate33 = cv2.dilate(img, ellipse33)
    # dilate53 = cv2.dilate(img, ellipse53)
    # dilate55 = cv2.dilate(img, ellipse55)
    # dilate55x2 = cv2.dilate(dilate55, ellipse55)
    # dilate55x3 = cv2.dilate(dilate55x2, ellipse55)

    opened53 = cv2.morphologyEx(img, cv2.MORPH_OPEN, ellipse53)
    closed53 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellipse53)

    distortions += [erode33, #erode53, erode55, erode55x2, 
                    dilate33, # dilate53, dilate55, dilate55x2,
                    opened53, closed53]

    return distortions

def shuffle_parallel(*args):
    for a in args: 
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)

def image_label_to_classes(img_label_list, 
                           preprocessor=preprocess_image, 
                           **kwargs): 
    res_list = []
    label_num_dict = {}
    num_label_dict = {}
    i = 1
    for img, label in img_label_list:
        p_img = preprocessor(img, **kwargs)
        if label not in label_num_dict: 
            label_num_dict[label] = i
            num_label_dict[i] = label
            i += 1
        res_list.append((p_img, label_num_dict[label]))

    return res_list, label_num_dict, num_label_dict

def distance_transform(img, 
                       metric=cv2.DIST_L2,
                       blocksize=5):

    return cv2.distanceTransform(img, 
                                 metric, 
                                 blocksize)

def distance_transform_label_list(img_label_list, 
                                  metric=cv2.DIST_L2, 
                                  blocksize=5): 

    return [(distance_transform(metric, blocksize), label) 
            for img, label in img_label_list]



def resize_components_vertical(img_label_list, 
                               sizey): 

    return [(cv2.resize(img, (img.shape[0], sizey)), label) 
            for img, label in img_label_list]

def resize_components_horizontal(img_label_list, sizex): 
    return [(cv2.resize(img, (sizex, img.shape[1])), label) 
            for img, label in img_label_list]

def crop_from_right(img, crop_size): 
    rows, cols = img.shape
    if cols > crop_size: 
        cropped = img[:, -1*crop_size:].copy()
        return cropped
    else: 
        resized = img.copy()
        resized.resize((img.shape[0], crop_size))
        return resized

def crop_from_left(img, crop_size): 
    rows, cols = img.shape
    if cols > crop_size: 
        cropped = img[:, :crop_size].copy()
        return cropped
    else: 
        resized = img.copy()
        resized.resize((resized.shape[0], crop_size))
        return resized

def resized_image_label_list(img_label_list, sizex, sizey): 
    return [(cv2.resize(img, (sizex, sizey)), label) for img, label in img_label_list]

def resized_image_label_list_to_x_train(img_label_list): 
    n_samples = len(img_label_list)
    img0 = img_label_list[0][0]
    if len(img0.shape) == 2: 
        n_channels = 1
        n_rows, n_cols = img0.shape
    else: 
        n_rows, n_cols, n_channels = img0.shape

    x_train = np.zeros(shape=(n_samples, n_channels, n_rows, n_cols), dtype=np.uint8)

    for i, img_class in enumerate(img_label_list): 
        img = img_class[0]
        x_train[i] = img

    return x_train

def image_to_profile_feature(img): 
    "Returns upper, lower profile and projection of an image"
    profile = np.zeros(shape=(3, 1, img.shape[1]), dtype=np.int)
    for i in range(img.shape[1]):
        pd = np.nonzero(img[:, i] > 0)[0]
        if len(pd) > 0:
            profile[0, 0, i] = min(pd) 
            profile[1, 0, i] = max(pd)
            profile[2, 0, i] = len(pd)
    return profile

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

def build_dataset(img_class_list, num_classes): 
    "Convert [(img, class)] list into x_train and y_train ndarrays for CNN"
    n_samples = len(img_class_list)
    LOG.debug("=== img_class_list[0] ===")
    LOG.debug(img_class_list[0])
    img_x, img_y = img_class_list[0][0].shape
    
    x_train = np.zeros(shape=(n_samples, 1, img_x, img_y), dtype=np.uint8)
    y_train = np.zeros(shape=(n_samples, ), dtype=np.uint8)

    for i, img_class in enumerate(img_class_list): 
        img = img_class[0]
        cls = img_class[1]
        x_train[i] = img
        y_train[i] = cls

    y_train = np_utils.to_categorical(y_train, num_classes + 1)
    return x_train, y_train

    

if __name__ == "__main__": 
    img_size = 64

    _image_files = aui.file_list(os.path.expandvars("$HOME/Datasets/leylamecnun/components-text-amiri/"))
    _image_file_list = aui.image_file_list(_image_files)
    _resized_file_list = resized_image_label_list(_image_file_list, img_size, img_size)
    x_train = resized_image_label_list_to_x_train(_resized_file_list)

