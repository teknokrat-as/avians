import os
import numpy as np
import cv2
import argparse
from datetime import datetime
import avians.util.image as aui

import logging
LOG = logging.getLogger(__name__)

NM_FILE_1 = os.path.expandvars('$HOME/Annex/Arabic/text-detection-arabic-models/td_trained_arbc.xml')
NM_FILE_2 = os.path.expandvars('$HOME/Annex/Arabic/text-detection-arabic-models/td_trained_arbc2.xml')

#Finds all channels and returns these
def find_channels(img):
    channels = cv2.text.computeNMChannels(img)
    cn = len(channels) - 1
    for c in range(0, cn):
        channels.append((255 - channels[c]))
    return channels

#Finds all rectangle points based on classifiers.
def find_text_regions(img, channels):
    all_points = []
    LOG.debug("=== channels ===")
    LOG.debug(channels)
    for channel in channels:
        erc1 = cv2.text.loadClassifierNM1(NM_FILE_1)
        er1 = cv2.text.createERFilterNM1(erc1,
                                         19,
                                         0.00015,
                                         0.13,
                                         0.2,
                                         True,
                                         0.1)
        erc2 = cv2.text.loadClassifierNM2(NM_FILE_2)
        er2 = cv2.text.createERFilterNM2(erc2, 0.5)
        regions = cv2.text.detectRegions(channel, er1, er2)
        if not regions:
            pass
        else:
            LOG.debug("=== regions ===")
            LOG.debug(regions)
            rects = cv2.text.erGrouping(img,channel,[r.tolist() for r in regions])
            LOG.debug("=== rects ===")
            LOG.debug(rects)
            for r in range(0,np.shape(rects)[0]):
                rect = rects[r]
                points = [rect[1], rect[1]+rect[3], rect[0], rect[0]+rect[2]]
                all_points.append(points)
    return all_points

def extract_regions(img, all_points):
    regions = np.zeros(shape=(len(all_points),
                              img.shape[0],
                              img.shape[1],
                              3),
                       dtype=img.dtype)
    for i, p in enumerate(all_points):
        reg = img[p[0]:p[1], p[2]:p[3]]
        regions[i, p[0]:p[1], p[2]:p[3]] = reg
    return regions

# Regionları extract etiğimiz image için rotate işlemi uyguluyoruz ve yeni pointleri elde ediyoruz

def rotated_rect_points(cnts):
    npoints = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        rect = cv2.minAreaRect(c)
        #print(rect)
        box = cv2.boxPoints(rect)
        #print(box)
        #print("*****")
        x1, y1, w1, h1 = cv2.boundingRect(box)
        points = [y1, y1+h1, x1, x1+w1]
        npoints.append(points)
    return npoints

def binarize_layers(layers):
    binarized = np.empty_like(layers)
    for i in range(layers.shape[0]):
        binarized[i] = otsu_binarize(layers[i])
    return binarized

def binarize_by_region(layers, regions):
    gray_layers = grayify_layers(layers)
    binarized_layers = np.zeros_like(gray_layers)
    inv_binarized_layers = np.zeros_like(gray_layers)
    for i, r in enumerate(regions):
        src = gray_layers[i, r[0]:r[1], r[2]:r[3]]
        dst = binarized_layers[i, r[0]:r[1], r[2]:r[3]]
        dst_inv = inv_binarized_layers[i, r[0]:r[1], r[2]:r[3]]
        cv2.threshold(src,
                      0,
                      255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                      dst=dst)
        cv2.threshold(src,
                      0,
                      255,
                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                      dst=dst_inv)
    return np.vstack((binarized_layers, inv_binarized_layers))


def grayify_layers(layers):
    if len(layers.shape) == 3:
        return layers
    else:
        grays = np.zeros(shape=(layers.shape[0],
                                layers.shape[1],
                                layers.shape[2]),
                         dtype=layers.dtype)
        for i in range(layers.shape[0]):
            grays[i] = cv2.cvtColor(layers[i], cv2.COLOR_BGR2GRAY)
        return grays

def text_object_segmentation(layers):
    all_results = []
    for i in range(layers.shape[0]):
        all_results.append(aui.text_image_segmentation(layers[i]))
    if len(all_results) > 0:
        tot_len = sum([ir.shape[0] for ir, sr in all_results])
        first_img = all_results[0][0]
        first_stat = all_results[0][1]
        all_images = np.empty(shape=(tot_len,
                                     first_img.shape[1],
                                     first_img.shape[2]),
                              dtype=first_img.dtype)
        all_stats = np.empty(shape=(tot_len,),
                             dtype=first_stat.dtype)
        ii = 0
        for ir, sr in all_results:
            li = ir.shape[0]
            ni = ii + li
            all_images[ii:ni] = ir[0:li]
            all_stats[ii:ni] = sr[0:li]
            ii = ni
        return all_images, all_stats
    else:  # if len(all_results) > 0
        return None, None

def find_text_components(img):
    ch = find_channels(img)
    all_regions = find_text_regions(img, ch)
    layers = extract_regions(img, all_regions)
    ## DEBUG
    # output_folder = '/tmp/text_detection_layers_' + datetime.now().strftime('%s')
    # os.makedirs(output_folder)
    # for i in range(layers.shape[0]):
    #     cv2.imwrite('{}/{}.png'.format(output_folder,
    #                                    i),
    #                 layers[i])
    ## END DEBUG
    binarized_layers = binarize_by_region(layers, all_regions)

    try:
        all_objects, all_stats = text_object_segmentation(binarized_layers)
    except MemoryError as me:
        all_objects, all_stats = None, None
        LOG.error("MemoryError for binarized layers with shape {}".format(binarized_layers.shape))
        output_folder = '/tmp/error_text_detection_binary_' + datetime.now().strftime('%s')
        os.makedirs(output_folder)
        for i in range(binarized_layers.shape[0]):
            cv2.imwrite('{}/{}.png'.format(output_folder,
                                           i),
                        binarized_layers[i])
        LOG.error("Wrote the layers to {}".format(output_folder)) 
    return all_objects, all_stats

def main_inner(image, library_dir, image_key):
    img = cv2.imread(image)
    # all_objects, all_stats = find_text_components(img)
    # output_folder = '/tmp/text_detection/{}/{}'.format(library_dir, image_key)
    # os.makedirs(output_folder)
    # for i in range(all_objects.shape[0]):
    #     cv2.imwrite('{}/{}.png'.format(output_folder,
    #                                    i),
    #                  all_objects[i])

    output_folder = os.path.expandvars('$HOME/Annex/Arabic/{}/{}/text_detection'.format(library_dir, image_key))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output = '{}/image.png'.format(output_folder)
    channels = find_channels(img)
    points = find_text_regions(img, channels)
    for p in points:
        cv2.rectangle(img, (p[2],p[0]), (p[3],p[1]), (0,255,0), 1)
    cv2.imwrite(output, img)
    return output

