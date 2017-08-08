import os
import argparse
import cv2
from datetime import datetime as dt
from avians.util.decorators import static_vars
import avians.util.image as aui
import avians.nn.common as anc
import avians.nn.models.component_cnn_v1 as anmccv1

import numpy as np
import sklearn.preprocessing as sp

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

def train_component_cnn(dataset,
                        img_x=64,
                        img_y=64,
                        n_channels=1,
                        batch_size=32,
                        nb_epoch=10):
    dataset = os.path.realpath(os.path.expandvars(dataset))
    if os.path.isfile(dataset): 
        ds = np.load(dataset)
        X = ds["X"]
        y = ds["y"]
    elif os.path.isdir(dataset): 
        X, y = produce_dataset(dataset, 
                               img_x, 
                               img_y,
                               n_channels=n_channels,
                               image_preprocessor=invert_and_resize_img)
        save_dataset(X=X, y=y)
    else: 
        raise Exception("{} is neither dir nor filename".format(dataset))

    n_labels = len(np.unique(y))
    n_samples, n_channel, img_rows, img_cols = X.shape
    print("Loaded {} with {} samples containing ({}x{}) pixels.".format(
        dataset, 
        n_samples, img_rows, img_cols, n_labels))

    le = sp.LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = anmccv1.train_model(X, y_encoded)
    anc.save_and_upload_model(model, save_dir=save_dir())
    return model

def produce_dataset(dataset_dir, img_x, img_y, n_channels, image_preprocessor): 
    image_label_list = anc.get_image_label_list(dataset_dir)
    n_samples = len(image_label_list)
    max_label_len = max([len(l) for img, l in image_label_list])
    X = np.empty(shape=(n_samples, img_x, img_y, n_channels),
                 dtype=np.uint8)
    y = np.empty(shape=(n_samples,),
                 dtype='S' + str(max_label_len))
    for i, img_lab in enumerate(image_label_list):
        img, lab = img_lab
        img_processed = image_preprocessor(img, img_x, img_y, n_channels)
        LOG.debug("=== img_processed.shape ===")
        LOG.debug(img_processed.shape)
        X[i] = np.atleast_3d(img_processed)
        y[i] = lab
    return X, y

def invert_and_resize_img(img, img_x, img_y, n_channels): 
    imgc = 255 - img
    imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY) if n_channels == 3 else imgc
    return cv2.resize(imgc, dsize=(img_x, img_y), interpolation=cv2.INTER_AREA)


def resize_img(img, img_x, img_y, n_channels):
    imgc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if n_channels == 1 else img
    return cv2.resize(imgc, dsize=(img_x, img_y), interpolation=cv2.INTER_AREA)

def save_dataset(**kwargs): 
    filename = os.path.join(save_dir(), "dataset.npz")
    np.savez_compressed(filename, 
                        **kwargs)

def letter_group_classes(img_label_list, sizex, sizey): 
    res_list = []
    label_num_dict = {}
    num_label_dict = {}
    i = 1
    for orig_img, label in img_label_list:
        # invert here
        img = huiu.invert_img(orig_img)
        distorted_imgs = hccu.distort_image(img)
        for img in distorted_imgs: 
            resized = hccu.preprocess_image(img, 
                                            invert=False, 
                                            sizex=sizex, 
                                            sizey=sizey, 
                                            dist_transform=True)
            if label not in label_num_dict: 
                label_num_dict[label] = i
                num_label_dict[i] = label
                i += 1
            res_list.append((resized, label_num_dict[label]))
    return res_list, label_num_dict, num_label_dict
