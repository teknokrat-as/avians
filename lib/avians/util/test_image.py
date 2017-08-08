
from nose.tools import with_setup
import avians.util.image as aui

import cv2
import numpy as np
import os

import logging
LOG = logging.getLogger(__name__)

image_file_list = []
IMAGE_DIR = os.path.expandvars("$HOME/Annex/Arabic/working-set-1/")


def setup_image_list():
    global image_file_list
    image_file_list = aui.file_list(IMAGE_DIR)

def teardown():
    pass

@with_setup(setup_image_list, teardown)
def test_enlarge_to_2x():
    global image_file_list
    LOG.debug("=== image_file_list ===")
    LOG.debug(image_file_list)
    
    for f in image_file_list:
        LOG.debug("=== f ===")
        LOG.debug(f)
        img = cv2.imread(f)
        img2x = aui.enlarge_and_filter(img, 2)
        assert img2x.shape[0] == img.shape[0] * 2
        assert img2x.shape[1] == img.shape[1] * 2

def test_generate_component_map(): 
    words = [("abcd", 4), ("ijkl", 6), ("şğuf", 5), ("ibadullah", 10)]

    for w, expected in words:
        w_img = aui.get_gray_word_image(w, fontname="Amiri", dpi=600,
                                        background=0,
                                        foreground=255)
        LOG.debug("=== w_img.shape ===")
        LOG.debug(w_img.shape)

        imgs, stats = aui.text_image_segmentation(w_img)
        component_map_1 = aui.generate_component_map(stats, PROXIMITY_BOUND=1)
        component_map_2 = aui.generate_component_map(stats, PROXIMITY_BOUND=2)

        assert component_map_1.shape[0] < component_map_2.shape[0]

def test_generate_component_graph(): 
    import matplotlib.pyplot as plt
    import networkx as nx    
    words = [("abcd", 4), ("ijkl", 6), ("şğuf", 5), ("ibadullah", 10)]

    for w, expected in words:
        w_img = aui.get_gray_word_image(w, fontname="Amiri", dpi=600,
                                        background=0,
                                        foreground=255)
        LOG.debug("=== w_img.shape ===")
        LOG.debug(w_img.shape)

        imgs, stats = aui.text_image_segmentation(w_img)
        component_map = aui.generate_component_map(stats, PROXIMITY_BOUND=0.5)
        graph = aui.component_map_to_nx(stats, component_map)
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        plt.savefig('/tmp/{}-graph.png'.format(w))
        plt.clf()
        assert nx.number_of_nodes(graph) == expected
        
def test_text_image_segmentation():

    words = [("abcd", 4), ("ijkl", 6), ("şğuf", 5), ("ibadullah", 10)]

    for w, expected in words:
        w_img = aui.get_gray_word_image(w, fontname="Amiri", dpi=600,
                                        background=0,
                                        foreground=255)
        LOG.debug("=== w_img.shape ===")
        LOG.debug(w_img.shape)

        imgs, stats = aui.text_image_segmentation(w_img)
        res_n, res_rows, res_cols = imgs.shape
        assert (res_rows, res_cols) == w_img.shape
        assert res_n == expected, "w: {}, res_n: {}, expected: {}".format(w, res_n, expected)

def test_decompose_and_resize():
    words = [("abcd", 4), ("ijkl", 6), ("şğuf", 5), ("ibadullah", 10)]

    for w, expected in words:
        w_img = aui.get_gray_word_image(w, fontname="Amiri", dpi=600,
                                        background=0,
                                        foreground=255)
        component_imgs = aui.decompose_and_resize(w_img, size_x=64, size_y=64)
        LOG.debug("=== component_imgs.shape ===")
        LOG.debug(component_imgs.shape)
        
        res_n, res_rows, res_cols = component_imgs.shape
        assert (res_rows, res_cols) == (64, 64)
        assert res_n == expected, "w: {}, res_n: {}, expected: {}".format(w, res_n, expected) 
        for j in range(res_n):
            c_img = component_imgs[j]
            bb = aui.img_bounding_box(c_img)
            LOG.debug("=== bb ===")
            LOG.debug(bb)
            if np.any(bb > 0):
                assert (bb[0, 0], bb[0, 1]) == (0, 64)
                assert (bb[1, 0], bb[1, 1]) == (0, 64)


def test_colors_of():
    aa = np.arange(60).reshape((4, 5, 3))
    clr = aui.colors_of(aa)
    LOG.debug("=== aa ===")
    LOG.debug(aa)
    LOG.debug("=== clr ===")
    LOG.debug(clr)
    
    for c in clr:
        assert c in aa
        
    for al in aa:
        for a in al: 
            assert a in clr, "{} cannot found in {}".format(a, clr)


def test_split_image():
    image_path = "$HOME/Annex/Arabic/working-set-1/rt-1-img8094.jpg"
    img = cv2.imread(os.path.expandvars(image_path))
    LOG.debug("=== img.shape before split ===")
    LOG.debug(img.shape)

    size = 32
    strides = 16
    image_arr = aui.split_image((img, image_path),
                                size,
                                size,
                                strides,
                                strides)

    n_split, size_x, size_y, n_ch = image_arr.shape
    assert size_x == size
    assert size_y == size
    LOG.debug("=== img.shape ===")
    LOG.debug(img.shape)
    LOG.debug("=== n_split, size_x, size_y, n_ch ===")
    LOG.debug((n_split, size_x, size_y, n_ch))
    LOG.debug("=== img.ndim ===")
    LOG.debug(img.ndim)

    assert n_ch == img.shape[2] if img.ndim == 3 else 1

    for i in range(n_split):
        cv2.imwrite("/tmp/split-image-res-{}.png".format(i), image_arr[i])


def test_split_image_list():
    image_path = "$HOME/Annex/Arabic/working-set-1/"
    files = aui.file_list(image_path)
    images = aui.image_file_list(files)
    size = 32
    strides = 16
    split_array = aui.split_image_list(images, size, size, strides, strides)
    n_split, size_x, size_y, n_ch = split_array.shape
    assert size_x == size
    assert size_y == size
    assert n_ch == 1


def test_color_quantization_mean_shift():
    image_path = "$HOME/Annex/Arabic/working-set-1/rt-1-img8094.jpg"
    img = cv2.imread(os.path.expandvars(image_path))
    centers, labels = aui.color_quantization_mean_shift(img)
    LOG.debug("=== centers ===")
    LOG.debug(centers)
    LOG.debug("=== labels ===")
    LOG.debug(labels)
    LOG.debug("=== len(labels) ===")
    LOG.debug(len(labels))
    LOG.debug("=== labels.shape ===")
    LOG.debug(labels.shape)
    LOG.debug("=== img.shape ===")
    LOG.debug(img.shape)

def test_color_quantization_image():
    image_path = "$HOME/Annex/Arabic/working-set-1/rt-1-img8094.jpg"
    img = cv2.imread(os.path.expandvars(image_path))
    img_cq = aui.color_quantization(img)
    cv2.imwrite("/tmp/img_cq.png", img_cq)


def test_color_layers():
    image_path = "$HOME/Annex/Arabic/working-set-1/rt-1-img8094.jpg"
    img = cv2.imread(os.path.expandvars(image_path))
    layers = aui.color_layers(img)
    for i, l in enumerate(layers):
        cv2.imwrite("/tmp/color-layer-{}.png".format(i), l)
