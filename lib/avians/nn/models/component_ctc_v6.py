from avians.nn.models import *

# Uncomment below for debugging
# import theano
# theano.config.exception_verbosity = "high"
# theano.config.optimizer = "None"

import os
import itertools
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input, Layer, Dense, Activation, Flatten
from keras.layers import Reshape, Lambda, merge, Permute, TimeDistributed
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks

import cv2 

import logging
LOG = logging.getLogger(__name__)

def add_noise(img):
    "Add salt/pepper and gaussian noise to the image to make it more realistic"

    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck

def read_dataset_labels_files(dataset_dir, f_regex=r'.*(png|jpg)$'): 
    "Reads image files from `dataset_dir` that conform to `f_regex`"

    label_list = os.listdir(dataset_dir)
    label_file_dict = {}
    for l in label_list: 
        ld = os.path.join(dataset_dir, l)
        label_file_dict[l] = [os.path.join(ld, f) 
                              for f in os.listdir(ld)
                              if re.match(f_regex, f)]
    return label_file_dict

def load_text_image(filename, w, h, bg=0):
    """Reads an image file and converts it to an image `w`x`h` pixels. It may also
change the background to `bg`."""

    img_orig = cv2.imread(filename, 0)
    img_orig = rotate_random(img_orig)
    orig_h , orig_w = img_orig.shape
    rh = h / orig_h if orig_h > h else 1
    rw = w / orig_w if orig_w > w else 1
    r = min(rw, rh)
    if r < 1: 
        img_res = cv2.resize(img_orig, dsize=None, fx=r, fy=r)
    else: 
        img_res = img_orig

    res_h, res_w = img_res.shape
    img_ret = np.ones(shape=(h, w),
                      dtype=np.uint8) * bg

    left = np.random.randint(h - res_h) if res_h < h else 0
    top = np.random.randint(w - res_w) if res_w < w else 0
    img_ret[left:(left + res_h),
            top:(top + res_w)] = img_res[:, :]
    img_ret //= 255
    img_ret = add_noise(img_ret)
    return img_ret

def create_blank_image(w, h, bg=0):
    "Returns a `w`x`h` blank image."
    img_ret = np.ones(shape=(w, h),
                      dtype=np.uint8) * bg
    img_ret = add_noise(img_ret)
    return img_ret.T

ARABIC_FONTS = aua.system_arabic_fonts()

def paint_text(text, w, h):
    "Prints an *Arabic* text on an image"
    font = np.random.choice(ARABIC_FONTS)
    arabic_text = aua.visenc_to_arabic(text)
    img = aui.get_gray_word_image(arabic_text,
                                  font,
                                  150,
                                  255,
                                  0,
                                  False)
    img_h, img_w = img.shape

    if (img_w < w) and (img_h < h):
        res_img = np.ones((h, w), np.int8) * 255
        top = np.random.randint(h - img_h)
        left = np.random.randint(w - img_w)
        # LOG.debug("=== (top, left) ===")
        # LOG.debug((top, left))
        res_img[top:(img_h + top), 
                left:(img_w + left)] = img
    else: 
        res_img = cv2.resize(img, dsize=(w, h))

    if LOG.isEnabledFor(logging.DEBUG): 
        cv2.imwrite("/dev/shm/{}-{}.png".format(text, 
                                                str(np.random.randint(10000))),
                    res_img)
    return res_img
                            
def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    "Shuffles the matrices or lists in `matrix_list` to use in CTC."
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('Only supports numpy.array and list objects')
    return ret

def text_to_labels_visenc(text): 
    "Converts a string of visenc in `text` to numeric labels to use in NN."
    global char_label_dict, char_label_index
    ret = []
    for c in text: 
        if c not in char_label_dict: 
            char_label_dict[c] = char_label_index
            char_label_index += 1
        ret.append(char_label_dict[c])
    return ret

def text_to_labels(text, num_classes):
    "Converts a string fo visenc in `text` to numeric label tuples."
    return list(tuple_label(text, num_classes))

def read_dataset_labels_files(dataset_dir, f_regex=r'.*(png|jpg)$'):
    """Reads filenames in `dataset_dir`, delimits to get label and filename and
returns a `dict[label] = [filenames]`"""
     
    label_list = os.listdir(dataset_dir)
    label_file_dict = {}
    for l in label_list: 
        ld = os.path.join(dataset_dir, l)
        label_file_dict[l] = [os.path.join(ld, f) 
                              for f in os.listdir(ld)
                              if re.match(f_regex, f)]
    return label_file_dict
    
def tuple_label(label, size=0):
    "Converts a list of strings to a tuple of ints"
    int_lst = []
    for l in label:
        int_lst.append(auad.VISENC_NUMERIC_DICT[l])
    if size > 0:
        if size > len(int_lst): 
            int_lst = int_lst[:size]
        else: 
            int_lst += [auad.VISENC_NUMERIC_DICT['']] * (size - len(int_lst))
    return tuple(int_lst)

def rotate_random(img, max_degrees=1):
    """Rotates `img` randomly to the right or left up to `max_degrees`"""
    rows, cols = img.shape
    deg = np.random.rand() * (max_degrees * 2) + max_degrees
    if deg < 0: 
        deg += 360
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
    dst = cv2.warpAffine(img, rotation_matrix, (cols, rows))
    return dst

## TODO: Write a Generator for items without labels.

class ClassificationImageGenerator(keras.callbacks.Callback):
    "In each iteration create a set of images from the given text images"
    def __init__(self, 
                 test_image_dir, 
                 minibatch_size,
                 img_w, 
                 img_h, 
                 downsample_width, 
                 val_split,
                 blank_label,
                 invert_loaded_images=False,
                 image_background=0,
                 absolute_max_string_len=64):
        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.test_image_dir = test_image_dir
        self.downsample_width = downsample_width
        self.val_split = val_split
        self.blank_label = blank_label
        self.absolute_max_string_len = absolute_max_string_len
        self.invert_loaded_images = invert_loaded_images
        self.file_list = aui.file_list(self.test_image_dir)
        self.image_background = image_background
        self.WORD_LIST_SIZE = 160000

    def get_output_size(self):
        return len(auad.VISENC_NUMERIC_DICT) + 1

    def build_arrays(self):
        self.num_words = len(self.file_list)
        self.max_string_len = self.absolute_max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.X_files =  []
        self.Y_len = [0] * self.num_words

        assert self.max_string_len <= self.absolute_max_string_len
        assert self.num_words % self.minibatch_size == 0
        assert (self.val_split * self.num_words) % self.minibatch_size == 0

        self.string_list = list(self.file_list)  # makes a copy
        if len(self.string_list) < self.num_words:
            self.string_list = self.string_list * (self.num_words // len(self.string_list) + 1)
        np.random.shuffle(self.string_list)
        self.string_list = self.string_list[:self.num_words]
        if len(self.string_list) != self.num_words:
            raise IOError('Could not pull enough files from dataset.')

        for i, f in enumerate(self.string_list):
            word = self.file_label_dict[f]
            self.Y_len[i] = len(word)
            ll = text_to_labels(word, self.get_output_size() + 1)
            self.Y_data[i, 0:len(word)] = ll
            self.X_text.append(word)
            self.X_files.append(f)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)
        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    def get_batch(self, index, size, train=False):
        X_data = np.ones([size, 1, self.img_h, self.img_w])
        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []

        for i in range(0, size):
            # Some blank inputs first to achieve translational invariance
            if train and i > size - 4:
                if K.image_dim_ordering() == 'th':
                    X_data[i, 0, :, :] = create_blank_image(self.img_w,
                                                            self.img_h,
                                                            self.image_background)
                else:
                    X_data[i, :, :, 0] = create_blank_image(self.img_w,
                                                            self.img_h,
                                                            self.image_background)
                labels[i, 0] = self.blank_label
                input_length[i] = self.downsample_width
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_dim_ordering() == 'th':
                    X_data[i, 0, :, :] = load_text_image(self.X_files[index + i],
                                                         self.img_w,
                                                         self.img_h,
                                                         self.image_background)
                else:
                    X_data[i, :, :, 0] = load_text_image(self.X_files[index + i],
                                                         self.img_w,
                                                         self.img_h,
                                                         self.image_background)
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.downsample_width
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])

        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index,
                                 self.minibatch_size,
                                 train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(self.WORD_LIST_SIZE)

    def on_epoch_begin(self, epoch, logs={}):
        self.build_word_list(self.num_words)

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, dataset_dir, minibatch_size,
                 img_w, img_h, downsample_width, val_split,
                 invert_loaded_images=False,
                 image_background=0,
                 absolute_max_string_len=64):
        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.dataset_dir = dataset_dir
        self.downsample_width = downsample_width
        self.val_split = val_split
        self.blank_label = max(list(auad.VISENC_NUMERIC_DICT.values())) + 1
        self.absolute_max_string_len = absolute_max_string_len
        self.invert_loaded_images = invert_loaded_images
        self.label_file_dict = read_dataset_labels_files(self.dataset_dir)
        self.file_list = []
        self.file_label_dict = {}
        for l, fl in self.label_file_dict.items(): 
            for f in fl: 
                self.file_list.append(f)
                self.file_label_dict[f] = l
            
        self.image_background = image_background

        self.WORD_LIST_SIZE = 160000

    def get_output_size(self):
        return len(auad.VISENC_NUMERIC_DICT) + 1

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words):
        self.num_words = num_words
        self.max_string_len = max([len(l) for l in self.label_file_dict])
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.X_files =  []
        self.Y_len = [0] * self.num_words

        assert self.max_string_len <= self.absolute_max_string_len
        assert self.num_words % self.minibatch_size == 0
        assert (self.val_split * self.num_words) % self.minibatch_size == 0

        self.string_list = list(self.file_list)  # makes a copy
        if len(self.string_list) < self.num_words:
            self.string_list = self.string_list * (self.num_words // len(self.string_list) + 1)
        np.random.shuffle(self.string_list)
        self.string_list = self.string_list[:self.num_words]
        if len(self.string_list) != self.num_words:
            raise IOError('Could not pull enough files from supplied dataset. ')

        for i, f in enumerate(self.string_list):
            word = self.file_label_dict[f]
            self.Y_len[i] = len(word)
            ll = text_to_labels(word, self.get_output_size() + 1)
            self.Y_data[i, 0:len(word)] = ll
            self.X_text.append(word)
            self.X_files.append(f)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)
        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    def get_batch(self, index, size, train):
        if K.image_dim_ordering() == 'th':
            X_data = np.ones([size, 1, self.img_h, self.img_w])
        else:
            X_data = np.ones([size, self.img_h, self.img_w, 1])
        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []

        for i in range(0, size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_dim_ordering() == 'th':
                    X_data[i, 0, :, :] = create_blank_image(self.img_w,
                                                            self.img_h,
                                                            self.image_background)
                else:
                    X_data[i, :, :, 0] = create_blank_image(self.img_w,
                                                            self.img_h,
                                                            self.image_background)
                labels[i, 0] = self.blank_label
                input_length[i] = self.downsample_width
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_dim_ordering() == 'th':
                    X_data[i, 0, :, :] = load_text_image(self.X_files[index + i],
                                                         self.img_w,
                                                         self.img_h,
                                                         self.image_background)
                else:
                    X_data[i, :, :, 0] = load_text_image(self.X_files[index + i],
                                                         self.img_w,
                                                         self.img_h,
                                                         self.image_background)
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.downsample_width
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])

        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': np.array(source_str)  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index,
                                 self.minibatch_size,
                                 train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(self.WORD_LIST_SIZE)

    def on_epoch_begin(self, epoch, logs={}):
        self.build_word_list(self.num_words)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    blank_label = max(list(auad.NUMERIC_VISENC_DICT.keys())) + 1
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            c = int(c)
            if c == blank_label:
                pass
            elif c in auad.NUMERIC_VISENC_DICT: 
                outstr += auad.NUMERIC_VISENC_DICT[c]
            else: 
                LOG.debug("=== c is not found in the dictionary ===")
                LOG.debug(c)
                LOG.debug("=== type(c) ===")
                LOG.debug(type(c))
                LOG.debug("=== out ===")
                LOG.debug(out)
                LOG.debug("=== out.shape ===")
                LOG.debug(out.shape)
                LOG.debug("=== out_best ===")
                LOG.debug(out_best)
                LOG.debug("=== outstr ===")
                LOG.debug(outstr)
                raise Exception
            # if c >= 0 and c < 26:
            #     outstr += chr(c + ord('a'))
            # elif c == 26:
            #     outstr += ' '
        ret.append(outstr)
    return ret

class VizCallback(keras.callbacks.Callback):

    def __init__(self, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, datetime.datetime.now().strftime('%A, %d. %B %Y %I.%M%p'))
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(0, num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % epoch))
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])

        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words, 1, i + 1)
            if K.image_dim_ordering() == 'th':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input, cmap='Greys_r')
            pylab.xlabel('Truth = \'%s\' Decoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 12)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % epoch))
        pylab.close()

from . import component_model as cm

class ComponentModelCTC(cm.AviansModel):
    def __init__(self, weight_file=None): 
        self.weight_file = weight_file
        self._set_model_params()
        self.create_model()

    def save_weights(self, output_file): 
        self.model.save_weights(output_file)
        
    def _set_model_params(self):
        # image params
        self.img_h = 64
        self.img_w = 512
        self.bg = 0
        # Network parameters
        self.conv_num_filters = 16
        self.filter_size = 3
        self.pool_size_1 = 4
        self.pool_size_2 = 2
        self.time_dense_size = 32
        self.rnn_size = 512
        self.time_steps = self.img_w / (self.pool_size_1 * self.pool_size_2)
        self.downsample_width = self.img_w / (self.pool_size_1 * self.pool_size_2) - 2
        self.output_size = len(auad.VISENC_NUMERIC_DICT) + 1
        self.input_shape = (1, self.img_h, self.img_w)
        self.absolute_max_string_len = 64
        self.blank_label = max(list(auad.NUMERIC_VISENC_DICT.keys())) + 1

    def create_model(self): 
        self._set_model_params()
        act = 'relu'
        input_data = Input(name='the_input', 
                           shape=self.input_shape,
                           dtype='float32')
        inner = Convolution2D(self.conv_num_filters,
                              self.filter_size,
                              self.filter_size,
                              border_mode='same',
                              activation=act,
                              name='conv1')(input_data)

        inner = MaxPooling2D(pool_size=(self.pool_size_1, self.pool_size_1),
                             name='max1')(inner)
        inner = Convolution2D(self.conv_num_filters,
                              self.filter_size,
                              self.filter_size,
                              border_mode='same',
                              activation=act,
                              name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.pool_size_2, 
                                        self.pool_size_2),
                             name='max2')(inner)
        conv_to_rnn_dims = (int((self.img_h / 
                                 (self.pool_size_1 * self.pool_size_2)) * 
                                self.conv_num_filters), 
                            int(self.img_w / 
                                (self.pool_size_1 * self.pool_size_2)))
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
        inner = Permute(dims=(2, 1), name='permute')(inner)
    
        # cuts down input size going into RNN:
        inner = TimeDistributed(Dense(self.time_dense_size,
                                      activation=act,
                                      name='dense1'))(inner)
    
        # Two layers of bidirecitonal GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(self.rnn_size,
                    return_sequences=True,
                    name='gru1')(inner)
        gru_1b = GRU(self.rnn_size,
                     return_sequences=True,
                     go_backwards=True,
                     name='gru1_b')(inner)
        gru1_merged = merge([gru_1, gru_1b], mode='sum')
        gru_2 = GRU(self.rnn_size,
                    return_sequences=True,
                    name='gru2')(gru1_merged)
        gru_2b = GRU(self.rnn_size,
                     return_sequences=True,
                     go_backwards=True)(gru1_merged)
    
        # transforms RNN output to character activations:
        inner = TimeDistributed(Dense(self.output_size, 
                                      name='dense2'))(
                                          merge([gru_2, gru_2b], mode='concat'))

        y_pred = Activation('softmax', name='softmax')(inner)
        # Model(input=[input_data], output=y_pred).summary()
        labels = Input(name='the_labels',
                       shape=[self.absolute_max_string_len],
                       dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func, 
                          output_shape=(1,),
                          name="ctc")([y_pred,
                                       labels,
                                       input_length,
                                       label_length])
        lr = 0.03
        # clipnorm seems to speeds up convergence
        clipnorm = 5
        sgd = SGD(lr=lr,
                  decay=3e-7,
                  momentum=0.9,
                  nesterov=True,
                  clipnorm=clipnorm)
        model = Model(input=[input_data,
                             labels,
                             input_length,
                             label_length], output=[loss_out])
        # model.summary()
        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        if self.weight_file is not None:
            model.load_weights(self.weight_file)

        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                      optimizer=sgd)
        self.model = model

        self._predictor = K.function([input_data], [y_pred])

        return model
    
    def prepare_binary_image(self, img_orig): 
        h = self.img_h
        w = self.img_w
        bg = self.bg
        img_orig = rotate_random(img_orig)
        orig_h, orig_w = img_orig.shape
        rh = h / orig_h if orig_h > h else 1
        rw = w / orig_w if orig_w > w else 1
        r = min(rw, rh)
        if r < 1: 
            img_res = cv2.resize(img_orig, dsize=None, fx=r, fy=r)
        else: 
            img_res = img_orig
        res_h, res_w = img_res.shape
        img_ret = np.ones(shape=(h, w),
                          dtype=np.uint8) * bg
        
        left = np.random.randint(h - res_h) if res_h < h else 0
        top = np.random.randint(w - res_w) if res_w < w else 0
        img_ret[left:(left + res_h),
                top:(top + res_w)] = img_res[:, :]
        img_ret //= 255
        if self.add_noise: 
            img_ret = add_noise(img_ret)
        return img_ret

    def _preprocess_image(self, img_orig): 
        LOG.debug("=== img_orig ===")
        LOG.debug(img_orig)
        orig_h, orig_w = img_orig.shape
        
        rh = self.img_h / orig_h if orig_h > self.img_h else 1
        rw = self.img_w / orig_w if orig_w > self.img_w else 1
        r = min(rw, rh)
        if r < 1: 
            img_res = cv2.resize(img_orig,
                                 dsize=None,
                                 fx=r,
                                 fy=r)
        else: 
            img_res = img_orig

        res_h, res_w = img_res.shape
        img_ret = np.ones(shape=(self.img_h, self.img_w),
                          dtype=np.uint8) * self.bg
        img_ret[:res_h,
                :res_w] = img_res[:, :]
        img_ret //= 255
        return img_ret

    def _preprocess_image_with_noise(self, img_orig): 
        h = self.img_h
        w = self.img_w
        bg = self.bg
        img_orig = rotate_random(img_orig)
        orig_h , orig_w = img_orig.shape
        rh = h / orig_h if orig_h > h else 1
        rw = w / orig_w if orig_w > w else 1
        r = min(rw, rh)
        if r < 1: 
            img_res = cv2.resize(img_orig, dsize=None, fx=r, fy=r)
        else: 
            img_res = img_orig

        res_h, res_w = img_res.shape
        img_ret = np.ones(shape=(h, w),
                          dtype=np.uint8) * bg

        left = np.random.randint(h - res_h) if res_h < h else 0
        top = np.random.randint(w - res_w) if res_w < w else 0
        img_ret[left:(left + res_h),
                top:(top + res_w)] = img_res[:, :]
        img_ret //= 255
        img_ret = add_noise(img_ret)
        return img_ret

    def _images_to_batch(self, image_arr,
                         stats,
                         sample_per_image=1):
        prep_func = self._preprocess_image_with_noise
        size = image_arr.shape[0]
        sample_size = size * sample_per_image
        X_data = np.ones([sample_size, 1, self.img_h, self.img_w])
        for ii in range(size): 
            img = image_arr[ii]
            s = stats[ii]
            img_view = img[s['x']:(s['x'] + s['w']),
                           s['y']:(s['y'] + s['h'])]
            for si in range(ii * sample_per_image, (ii + 1) * sample_per_image):
                X_data[si, 0, :, :] = prep_func(img_view)
        
        labels = np.zeros([size, self.absolute_max_string_len])
        input_length = np.ones([size, 1]) * self.downsample_width
        label_length = np.zeros([size, 1])
        source_str = []

        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        
        return (inputs, outputs)

    def _image_to_word_batch(self, images):
        size = len(images)
        X_data = np.ones([size, 1, self.img_h, self.img_w])
        labels = np.zeros([size, self.absolute_max_string_len])
        input_length = np.ones([size, 1]) * self.downsample_width
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(0, size):
            X_data[i, 0, :, :] = images[i]
       
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        
        return (inputs, outputs)

    def train_model(self, 
                    dataset_dir, 
                    nb_epoch=10,
                    words_per_epoch=16000):
        LOG.debug("=== dataset_dir ===")
        LOG.debug(dataset_dir)
        
        model = self.create_model()
        val_split = 0.2
        val_words = int(words_per_epoch * (val_split))
        img_gen = TextImageGenerator(dataset_dir=dataset_dir,
                                     minibatch_size=32,
                                     img_w=self.img_w,
                                     img_h=self.img_h,
                                     downsample_width=self.img_w / (self.pool_size_1 * self.pool_size_2) - 2,
                                     val_split=words_per_epoch - val_words)
        viz_cb = VizCallback(self._predictor, img_gen.next_val())
        model.fit_generator(generator=img_gen.next_train(),
                            samples_per_epoch=(words_per_epoch - val_words),
                            nb_epoch=nb_epoch, 
                            validation_data=img_gen.next_val(), 
                            nb_val_samples=val_words,
                            callbacks=[viz_cb, img_gen])

        return model


    def _pred_output_to_labels(self,
                               out): 
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            LOG.debug("out_best: {}".format(out_best))
            LOG.debug("out_best w/o blank: {}".format([o for o in out_best if o != self.blank_label]))
            outstr = ''
            for c in out_best:
                c = int(c)
                if c == self.blank_label:
                    pass
                elif c in auad.NUMERIC_VISENC_DICT: 
                    outstr += auad.NUMERIC_VISENC_DICT[c]
                else: 
                    LOG.debug("=== c is not found in the dictionary ===")
                    LOG.debug(c)
                    LOG.debug("=== type(c) ===")
                    LOG.debug(type(c))
                    LOG.debug("=== out ===")
                    LOG.debug(out)
                    LOG.debug("=== out.shape ===")
                    LOG.debug(out.shape)
                    LOG.debug("=== out_best ===")
                    LOG.debug(out_best)
                    LOG.debug("=== outstr ===")
                    LOG.debug(outstr)
                    raise Exception
            LOG.debug("=== (out_best, outstr) ===")
            LOG.debug((",".join([str(o) for o in out_best if o != self.blank_label]), 
                       outstr))
            ret.append(outstr)
        return ret

    def _classify_text_objects(self,
                               images,
                               stats,
                               total_votes,
                               max_labels_per_image): 
        n_images = images.shape[0]
        word_batch = self._images_to_batch(images,
                                           stats,
                                           sample_per_image=total_votes)
        
        pred_input = word_batch[0]["the_input"]
        LOG.debug("pred_input.shape: {}".format(pred_input.shape))
        out = self._predictor([pred_input])[0]
        LOG.debug("=== out.shape ===")
        LOG.debug(out.shape)
        labels = self._pred_output_to_labels(out)
        label_res = np.zeros(dtype=[('label', 'S64'),
                                    ('prob', np.float)],
                             shape=(n_images, max_labels_per_image))

        for ii in range(n_images): 
            votes = {}
            for li in range(ii * total_votes,
                            (ii + 1) * total_votes): 
                l = labels[li]
                if l not in votes: 
                    votes[l] = 0
                votes[l] += 1
            sorted_votes = [(k, votes[k] / total_votes) 
                            for k in sorted(votes, 
                                            key=votes.get,
                                            reverse=True)]
            for i, l in enumerate(sorted_votes[:max_labels_per_image]):
                label_res[ii, i]['label'] = l[0]
                label_res[ii, i]['prob'] = l[1]

        return label_res
        
    def classify_text_objects(self,
                              text_objects,
                              total_votes=100,
                              max_labels_per_image=10,
                              max_size_per_batch=1000):
        images = text_objects[0]
        stats = text_objects[1]
        n_images = images.shape[0]
        label_res_list = []
        max_objects_per_batch = max_size_per_batch // total_votes
        for begin in range(0, n_images, max_objects_per_batch):
            end = begin + max_objects_per_batch
            lr = self._classify_text_objects(images[begin:end],
                                             stats[begin:end],
                                             total_votes,
                                             max_labels_per_image)
            label_res_list.append(lr)

        return np.concatenate(label_res_list)

    def classify_image(self, image, total_votes=100):
        prepared = []
        for i in range(total_votes):
            prepared.append(self._preprocess_image_with_noise(image))
        word_batch = self._image_to_word_batch(prepared)
        pred_input = word_batch[0]["the_input"]
        cv2.imwrite("/dev/shm/pred-input-{}.png".format(
            np.random.randint(999999)),
            pred_input[0, 0, :, :] * 255)
        LOG.debug("pred_input.shape: {}".format(pred_input.shape))
        out = self._predictor([pred_input])[0]
        LOG.debug("out: {}".format(out))
        ret = self._pred_output_to_labels(out)
        
        votes = {}
        for s in ret: 
            if s not in votes: 
                votes[s] = 0
            votes[s] += 1
        sorted_votes = [(k, votes[k] / total_votes) for k in sorted(votes, 
                                                      key=votes.get,
                                                      reverse=True)]
        return sorted_votes

    def load_text_image(self, filename):
        h = self.img_h
        w = self.img_w
        bg = self.bg
        img_orig = cv2.imread(filename, 0)
        img_orig = rotate_random(img_orig)
        orig_h, orig_w = img_orig.shape
        rh = h / orig_h if orig_h > h else 1
        rw = w / orig_w if orig_w > w else 1
        r = min(rw, rh)
        if r < 1: 
            img_res = cv2.resize(img_orig, dsize=None, fx=r, fy=r)
        else: 
            img_res = img_orig
        res_h, res_w = img_res.shape
        img_ret = np.ones(shape=(h, w),
                          dtype=np.uint8) * bg
        
        left = np.random.randint(h - res_h) if res_h < h else 0
        top = np.random.randint(w - res_w) if res_w < w else 0
        img_ret[left:(left + res_h),
                top:(top + res_w)] = img_res[:, :]
        img_ret //= 255
        img_ret = add_noise(img_ret)
        return img_ret

        
    def preprocess_images(self, file_list):
        LOG.debug("=== len(file_list) ===")
        LOG.debug(len(file_list))
        # image_list = aui.image_file_list(file_list)
        # LOG.debug("=== len(image_list) ===")
        # LOG.debug(len(image_list))
        preprocessed_list = [self.load_text_image(f) 
                             for f in file_list]
        LOG.debug("=== len(preprocessed_list) ===")
        LOG.debug(len(preprocessed_list))
        size = len(preprocessed_list)
        X_data = np.ones([size, 1, self.img_h, self.img_w])
        LOG.debug("=== X_data.shape ===")
        LOG.debug(X_data.shape)

        for i, c in enumerate(preprocessed_list): 
            X_data[i] = c
            if LOG.isEnabledFor(logging.DEBUG): 
                cv2.imwrite('/dev/shm/avians-ctc-v6-{}.png'.format(i),
                            c * 255)

        input_length = np.ones([size, 1]) * self.downsample_width

        inputs = {'the_input': X_data,
                  'the_labels': None,
                  'input_length': input_length,
                  'label_length': None
                  }

        self.inputs = inputs

        return inputs
