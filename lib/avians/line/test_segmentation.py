

import harfbin.line.segmentation as hls
import harfbin.util.image_utils as huiu
from scipy import ndimage
from nose.tools import with_setup
import numpy as np
import cv2

def get_text_component_array(): 
    w_img = huiu.get_gray_word_image("harfzen", 
                                     fontname="Amiri", 
                                     dpi=300, 
                                     background=0, 
                                     foreground=255)

    w_arr = huiu.text_image_segmentation(w_img)
    return w_arr


def test_find_lines_by_box(): 
    text = """رباعی

ای نشأه حسنی عشقه تأثیر قیلن 
عشقیله بناء كونی تعمیر قیلن
لیلی سر زلفنی گره گیر قیلن
مجنون حزین بویننه زنجیر قیلن
"""

    for dpi in [150, 300, 600, 900, 1200]: 
        text_img = huiu.get_gray_word_image(text, 
                                            fontname="Amiri", 
                                            dpi=dpi, 
                                            background=0, 
                                            foreground=255)
        rot_img = ndimage.rotate(text_img, 30)
    
        line_imgs = hls.find_lines_by_box(rot_img, 1.5)
        assert line_imgs.shape[0] <= 5, "For dpi: {} Shape: {}".format(dpi, line_imgs.shape)


def test_find_lines_by_dilate(): 
    text = """رباعی

ای نشأه حسنی عشقه تأثیر قیلن 
عشقیله بناء كونی تعمیر قیلن
لیلی سر زلفنی گره گیر قیلن
مجنون حزین بویننه زنجیر قیلن
"""

    for dpi in [150, 300, 600, 900, 1200]: 
        text_img = huiu.get_gray_word_image(text, 
                                            fontname="Amiri", 
                                            dpi=dpi, 
                                            background=0, 
                                            foreground=255)
        rot_img = ndimage.rotate(text_img, 30)
    
        line_imgs = hls.find_lines_by_dilate(rot_img, 0.4)
        assert line_imgs.shape[0] < 5, "For dpi: {} Shape: {}".format(dpi, line_imgs.shape)
    
def test_dilate_array():
    w_arr = get_text_component_array()
    d_arr = hls.dilate_array(w_arr)
    for i in range(w_arr.shape[0]): 
        assert d_arr[i] == huiu.dilate(w_arr[i])
        
def test_merge_image_array():
    w_arr = get_text_component_array()
    merged = hls.merge_image_array(w_arr)
    assert merged == w_arr

def test_minimum_rectangles():
    MAX_IMG_SIZE=1000
    imgs = np.zeros(shape=(MAX_IMG_SIZE/10,
                           MAX_IMG_SIZE, 
                           MAX_IMG_SIZE),
                    dtype=np.uint8)
    
    for i in range(MAX_IMG_SIZE//10):
        n = i * 5 + 1
        points = np.array([[[0, n], [n, 0], [2 * n, n], [n, 2 * n]]],
                          dtype=np.int32)
        print(imgs.shape)
        print(imgs[i, :, :].shape)
        print(points)
        cv2.polylines(imgs[i, :, :], points, True, 255, 1)
    
    min_rect_arr = hls.minimum_rectangles(imgs)
    redrawed_min_rectangles = hls.draw_min_rectangles(min_rect_arr)
    assert redrawed_min_rectangles == imgs
    
def test_mask_image():
    masks = np.zeros(shape=(4, 100, 100), dtype=np.int8)
    masks[0, :50, :50] = 1
    masks[1, 50:, :50] = 1
    masks[2, 50:, 50:] = 1
    masks[3, :50, 50:] = 1
    
    img = np.ones(shape=(100, 100), dtype=np.int8)

    masked = hls.mask_image(img, masks)
    assert np.all(masked == masks)
