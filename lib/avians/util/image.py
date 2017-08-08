# from PIL import Image
# from PIL import ImageFont
# from PIL import ImageDraw
# import bidi.algorithm as ba
import cv2
import numpy as np
import numpy.linalg as nl
import uuid
import os
import re
import itertools
import gc

import networkx as nx
import time

import numpy.lib.recfunctions as nlr
import sklearn.cluster as sc

import logging
LOG = logging.getLogger(__name__)

ADAPTIVE_GAUSSIAN_BLOCK_SIZE = 101
GAUSSIAN_BLUR_WINDOW = (3, 3)
GAUSSIAN_BLUR_SIGMAX = 0.3

DISTANCE_TRANSFORM_METRIC = cv2.DIST_L2
DISTANCE_TRANSFORM_BLOCKSIZE = 5


def invert_img(img):
    return (255 - img).astype(img.dtype)


def file_list(dataset_path, f_regex=r'.*(png|jpg)$'):
    dataset_dir = os.path.expandvars(dataset_path)
    if os.path.isdir(dataset_dir):
        return [os.path.join(dataset_dir, f)
                for f in os.listdir(dataset_dir)
                if re.match(f_regex, f)]
    else:
        # Return a single file
        return [dataset_dir]


def recursive_file_list(dataset_dir, f_regex=r'.*(png|jpg)$'):
    all_files = []
    dataset_dir = os.path.expandvars(dataset_dir)
    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        all_files += [os.path.join(root, name)
                      for name in files if re.match(f_regex, name)]
    return all_files


def get_artifact_dir(filename):
    dd = os.path.splitext(filename)[0]
    if not os.path.isdir(dd):
        os.mkdir(dd)
    return dd


def image_file_list(file_list, imread_flags=0):
    return [(cv2.imread(f, imread_flags), f) for f in file_list]


def dilate(img, times=1, size=(3, 3)):
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    res = img
    for i in range(times):
        res = cv2.dilate(res, ellipse)
    return res


def erode(img, times=1, size=(3, 3)):
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    res = img
    for i in range(times):
        res = cv2.erode(res, ellipse)
    return res


def erode_dilate(img):
    "Apply erode + dilate to the image, then send it to get_contours"

    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_img = cv2.erode(img, ellipse)
    dilated_eroded_img = cv2.dilate(eroded_img, ellipse)
    return dilated_eroded_img


def enlarge_and_filter(img, resize_factor=2):
    assert resize_factor > 1
    resized = cv2.resize(img, dsize=None, fx=resize_factor, fy=resize_factor)
    filtered = cv2.bilateralFilter(
        resized, d=-1, sigmaSpace=resize_factor * 2, sigmaColor=100)
    return filtered


def enlarge_and_get_contours(img, min_size=4000):
    max_current = max(img.shape[0], img.shape[1])

    if max_current < min_size:
        resize_factor = int(np.ceil(min_size / max_current))
        resized = cv2.resize(img, dsize=(
            0, 0), fx=resize_factor, fy=resize_factor)
    else:
        resized = img
    res_contours = contours(resized)
    return resized, res_contours


def otsu_binarize(img, invert=False):
    if invert:
        ot, th1 = cv2.threshold(img,
                                0,
                                255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        ot, th1 = cv2.threshold(img,
                                0,
                                255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th1


def adaptive_mean(img, max_value=-1, invert=False, block_size=11, c_const=0):
    # img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    max_value = np.max(np.max(img[:])) if max_value == -1 else max_value
    invert_param = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(img, max_value, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 invert_param, block_size, c_const)


def adaptive_gaussian(img, max_value=-1, invert=False, block_size=11, c_const=0):
    # img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    max_value = np.max(np.max(img[:])) if max_value == -1 else max_value
    invert_param = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(img, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 invert_param, block_size, c_const)


def color_quantization_kmeans(img, n_colors=10, attempts=10):
    img_arr = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                attempts, 1.0)
    compactness, label, center = cv2.kmeans(img_arr,
                                            n_colors,
                                            None,
                                            criteria,
                                            n_colors,
                                            cv2.KMEANS_RANDOM_CENTERS)

    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res2 = res.reshape((img.shape))
    # res3 = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)
    return center, label


def color_quantization_mean_shift(img):
    img_arr = np.float32(img.reshape((-1, 3)))
    bandwidth = sc.estimate_bandwidth(img_arr, quantile=0.2, n_samples=500)
    if np.isclose(bandwidth, 0):
        return None, None

    LOG.debug("=== bandwidth ===")
    LOG.debug(bandwidth)
    ms = sc.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(img_arr)
    centers = ms.cluster_centers_
    labels = ms.labels_
    return centers, labels


def color_layers(img):
    centers, labels = color_quantization_mean_shift(img)
    unique_labels = np.unique(labels)
    layers = np.zeros(shape=(len(unique_labels), img.shape[0],
                             img.shape[1]),
                      dtype=img.dtype)
    masks = labels.reshape((img.shape[0], img.shape[1]))
    for i, l in enumerate(unique_labels):
        layers[i][masks == l] = 255
    return layers


def color_quantization(img_orig,
                       n_colors=5):
    LOG.debug("=== ***color_quantization*** ===")

    img = cv2.GaussianBlur(img_orig,
                           GAUSSIAN_BLUR_WINDOW,
                           GAUSSIAN_BLUR_SIGMAX)
    # centers, labels = color_quantization_mean_shift(img)
    # if not np.any(centers):
    centers, labels = color_quantization_kmeans(img,
                                                n_colors)

    LOG.debug("=== centers ===")
    LOG.debug(centers)
    LOG.debug("=== labels ===")
    LOG.debug(labels)

    colors = np.uint8(centers)
    LOG.debug("=== colors ===")
    LOG.debug(colors)

    masks = labels.reshape((img.shape[0], img.shape[1]))
    n_colors, n_ch = colors.shape
    out_arr = np.empty(shape=(n_colors, img.shape[0], img.shape[1], n_ch),
                       dtype=img.dtype)

    for i in range(n_colors):
        cpy = np.zeros_like(img)
        cpy[masks == i] = colors[i]
        out_arr[i] = cpy
    return out_arr


def contours_via_color_quantization(img, n_colors=10):

    quantized, centers = color_quantization(img, n_colors)
    all_contours = []
    for c in centers:
        cimg = np.uint8(quantized == c) * 255
        cimg = np.array(cimg[:, :, 0])
        rimg, contours, h = cv2.findContours(cimg,
                                             cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_NONE)
        all_contours += contours
    return all_contours


def filter_small_contours(contours, threshold):
    boxes = [cv2.boundingRect(c) for c in contours]
    return [c for b, c in zip(boxes, contours) if b[2] * b[3] > threshold]


def contours(img):
    if len(img.shape) == 2:
        gray_img = img
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    LOG.debug("=== gray_img.shape ===")
    LOG.debug(gray_img.shape)

    # framed = copy_with_frame(gray_img, 2)
    # blurred = cv2.GaussianBlur(gray_img,
    #                            GAUSSIAN_BLUR_WINDOW,
    #                            GAUSSIAN_BLUR_SIGMAX)
    # binarized = adaptive_gaussian(blurred,
    # block_size=ADAPTIVE_GAUSSIAN_BLOCK_SIZE)
    # erode_dilate_img = erode_dilate(binarized)

    img, img_contours, hierarchy = cv2.findContours(gray_img,
                                                    cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_NONE)
    LOG.debug("=== len(img_contours) ===")
    LOG.debug(len(img_contours))

    contour_tree = get_contour_tree(img_contours, hierarchy, 1)
    LOG.debug("=== len(contour_tree) ===")
    LOG.debug(len(contour_tree))
    contour_and_holes = get_contours_and_holes(contour_tree)
    LOG.debug("=== len(contour_and_holes) ===")
    LOG.debug(len(contour_and_holes))
    return contour_and_holes


def contours_and_stats(gray_img):
    img, contours, hier = cv2.findContours(gray_img,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    n_contours = len(contours)
    components = np.zeros(shape=(n_contours,
                                 gray_img.shape[0],
                                 gray_img.shape[1]),
                          dtype=gray_img.dtype)
    stats = np.empty(shape=(n_contours,),
                     dtype=[("x", np.int32),
                            ("y", np.int32),
                            ("x2", np.int32),
                            ("y2", np.int32),
                            ("w", np.int32),
                            ("h", np.int32),
                            ("area", np.int32),
                            ("level", np.int32),
                            ("n_children", np.int32),
                            ("parent", np.int32)])
    stats['parent'] = -1
    # Draw outer edges
    for ci in range(n_contours):
        cv2.drawContours(components[ci], contours, ci, 255, -1)
        stats['x'][ci] = contours[ci][:, :, 1].min()
        stats['y'][ci] = contours[ci][:, :, 0].min()
        stats['x2'][ci] = contours[ci][:, :, 1].max()
        stats['y2'][ci] = contours[ci][:, :, 0].max()
        stats['parent'][ci] = hier[0, ci, 3]

    # Draw holes on parent and calculate area
    for ci in range(n_contours):
        if stats['parent'][ci] >= 0:
            pi = stats['parent'][ci]
            cv2.drawContours(components[pi], contours, ci, 0, -1)
            stats['level'][ci] = stats['level'][pi] + 1
            stats['n_children'][pi] += 1
        else:
            stats['level'][ci] = 0

        stats['area'][ci] = components[ci].nonzero()[0].shape[0]

    stats['w'] = stats['x2'] - stats['x']
    stats['h'] = stats['y2'] - stats['y']

    return components, stats


def get_contours_and_holes(contour_tree):
    """Flattens the contour tree into a (contour, holes, level) structure. 

    If a contour at level n has holes at level n-1, which in turn has holes at
    n-2, contour and its holes at n-1 are added as elements. 

    """
    contours_holes = []

    while len(contour_tree) > 0:
        cnt, children, level = contour_tree[0]
        holes = []
        for ch in children:
            # get only contours as holes
            hole, grandchildren, ll = ch
            holes.append(hole)
            # add children as elements to contour tree
            contour_tree.append(ch)
        # create a contours_holes element

        ch = (cnt, holes, level)
        contours_holes.append(ch)
        del contour_tree[0]

    return contours_holes


def copy_with_frame(img, frame_size=1):
    if len(img.shape) == 2:
        res = np.ones((img.shape[0] + frame_size * 2,
                       img.shape[1] + frame_size * 2),
                      img.dtype) * 255
    elif len(img.shape) == 3:
        res = np.ones((img.shape[0] + frame_size * 2,
                       img.shape[1] + frame_size * 2,
                       img.shape[2]), img.dtype) * 255

    res[frame_size:(-1 * frame_size), frame_size:(-1 * frame_size)] = img[:]
    return res


def distance_transform(img):
    return cv2.distanceTransform(img,
                                 DISTANCE_TRANSFORM_METRIC,
                                 DISTANCE_TRANSFORM_BLOCKSIZE)


def get_contour_tree(contours, hierarchy, next_contour_idx):
    """Generates a hierarchical list of tuples

    Each item is contours[i] = (contour, children_dict, color)

    where the contour is an array, children_dict is the same structure as
    contours, color is a grayscale integer to paint this contour in the image.

    This function recursively returns this structure.

    """
    contour_tree = []
    while len(contours) > 1 and next_contour_idx != -1:
        # LOG.debug("=== current_contour_idx ===")
        # LOG.debug(next_contour_idx)
        current_contour = contours[next_contour_idx]
        # LOG.debug("=== len(current_contour) ===")
        # LOG.debug(len(current_contour))

        # # LOG.debug("=== hierarchy ===")
        # # LOG.debug(hierarchy)

        # LOG.debug("=== hierarchy.shape ===")
        # LOG.debug(hierarchy.shape)

        child_contour_idx = hierarchy[next_contour_idx, 2]
        # LOG.debug("=== child_contour_idx ===")
        # LOG.debug(child_contour_idx)

        children = get_contour_tree(contours, hierarchy, child_contour_idx)
        # # LOG.debug("=== children ===")
        # # LOG.debug(children)
        if len(children) > 0:
            level = max([c[2] for c in children]) + 1
            # LOG.debug("=== [c[2] for c in children] ===")
            # LOG.debug([c[2] for c in children])
        else:
            level = 1
        contour_tree.append((current_contour, children, level))
        # LOG.debug("=== len(contour_tree) ===")
        # LOG.debug(len(contour_tree))

        next_contour_idx = hierarchy[next_contour_idx, 0]
        # LOG.debug("=== next_contour_idx ===")
        # LOG.debug(next_contour_idx)

    # LOG.debug("=== len(contour_tree) returns ===")
    # LOG.debug(len(contour_tree))

    return contour_tree


def hex_color(rgb_tuple):
    return (hex(rgb_tuple[0])[2:] +
            hex(rgb_tuple[1])[2:] +
            hex(rgb_tuple[2])[2:])


def text_image_segmentation(gray_img,
                            min_area=0,
                            max_area=1.0):
    "Returns each connected component in its own image stacked on an array"
    gray_copy = gray_img.copy()
    LOG.debug("=== gray_copy.shape ===")
    LOG.debug(gray_copy.shape)
    LOG.debug("=== gray_copy.dtype ===")
    LOG.debug(gray_copy.dtype)

    n_label, label_img, stats, centroids = cv2.connectedComponentsWithStats(
        gray_copy)
    LOG.debug("=== label_img.dtype ===")
    LOG.debug(label_img.dtype)
    LOG.debug("=== label_img.shape ===")
    LOG.debug(label_img.shape)
    LOG.debug("=== n_label ===")
    LOG.debug(n_label)
    LOG.debug("=== label_img ===")
    LOG.debug(label_img)

    if LOG.isEnabledFor(logging.DEBUG):
        cv2.imwrite("/tmp/label_img-{}.png".format(np.random.randint(10000)),
                    label_img)

        # np.savez_compressed("/tmp/debug-array-{}.npy".format(
        #     np.random.randint(10000)),
        #                     label_img=label_img,
        #                     gray_img=gray_img,
        #                     gray_copy=gray_copy)
    # if gray_img.shape[1] > 1000:
    #     assert False

    gc.collect()
    result_shape = (n_label, gray_img.shape[0], gray_img.shape[1])
    result_arr = np.zeros(shape=result_shape,
                          dtype=gray_img.dtype)
    LOG.debug("=== label_img.max() ===")
    LOG.debug(label_img.max())

    assert label_img.max() == (n_label - 1)
    for i in range(n_label):
        g_idx = (label_img == i)
        g_cpy = np.zeros_like(gray_img)
        g_cpy[g_idx] = 255  # gray_img[g_idx]
        # if LOG.isEnabledFor(logging.DEBUG):
        #     cv2.imwrite("/tmp/g_cpy-{}.png".format(i), g_cpy)
        result_arr[i] = g_cpy

    stats_arr = stats.view(dtype=[("y", stats.dtype),
                                  ("x", stats.dtype),
                                  ("h", stats.dtype),
                                  ("w", stats.dtype),
                                  ("area", stats.dtype)])
    stats_arr = nlr.append_fields(stats_arr,
                                  ["cy", "cx"],
                                  [centroids[:, 0], centroids[:, 1]])
    # WARNING: The first element of the component array is the inverted
    # original image, we remove it here
    result_arr = result_arr[1:]
    stats_arr = stats_arr[1:]

    total_area = gray_img.shape[0] * gray_img.shape[1]
    LOG.debug("=== total_area ===")
    LOG.debug(total_area)
    if max_area <= 1 and max_area > 0:
        max_area = total_area * max_area
    if min_area <= 1 and min_area > 0:
        min_area = total_area * min_area

    LOG.debug("=== max_area ===")
    LOG.debug(max_area)
    LOG.debug("=== min_area ===")
    LOG.debug(min_area)
    LOG.debug("=== stats_arr['area'].mean() ===")
    LOG.debug(stats_arr['area'].mean())
    LOG.debug("=== stats_arr['area'].std() ===")
    LOG.debug(stats_arr['area'].std())
    selection = np.bitwise_and(stats_arr['area'] >= min_area,
                               stats_arr['area'] <= max_area)
    LOG.debug("=== selection ===")
    LOG.debug(selection)
    LOG.debug("=== ult_arr.nbytes ===")
    LOG.debug(result_arr.nbytes)
    
    # [h.flush() for h in LOG.handlers]
    # time.sleep() 
    # result_arr = result_arr[selection]

    # stats_arr = stats_arr[selection]
    return result_arr[selection], stats_arr[selection]

def image_segmentation(img,
                       do_color_quantization=True,
                       do_invert=False,
                       do_dilate=0,
                       min_area=0.0,
                       max_area=1.0):
    LOG.debug("=== ***image_segmentation*** ===")
    img = np.atleast_3d(img)

    if do_color_quantization:
        color_layers = color_quantization(img)
    else:
        color_layers = np.empty(shape=(1,
                                       img.shape[0],
                                       img.shape[1],
                                       img.shape[2]),
                                dtype=img.dtype)
        color_layers[0] = img

    # if LOG.isEnabledFor(logging.DEBUG):
    #     np.savez_compressed("/tmp/color-layers-{}".format(
    #         np.random.randint(1000)), color_layers=color_layers)
    if do_invert and (do_dilate > 0):
        white_layers_shape = (color_layers.shape[0] * 4,
                              color_layers.shape[1],
                              color_layers.shape[2])

    elif do_invert or (do_dilate > 0):
        white_layers_shape = (color_layers.shape[0] * 2,
                              color_layers.shape[1],
                              color_layers.shape[2])
    else:
        white_layers_shape = (color_layers.shape[0],
                              color_layers.shape[1],
                              color_layers.shape[2])

    white_layers = np.empty(shape=white_layers_shape,
                            dtype=np.uint8)

    for i in range(color_layers.shape[0]):
        cpy = np.zeros(shape=(color_layers.shape[1],
                              color_layers.shape[2]),
                       dtype=np.uint8)
        nonzero = np.bitwise_or(color_layers[i, :, :, 0] > 0,
                                color_layers[i, :, :, 1] > 0,
                                color_layers[i, :, :, 2] > 0)
        cpy[nonzero] = 255
        white_layers[i] = cpy

    last_pos = color_layers.shape[0]

    if do_invert:
        for i in range(color_layers.shape[0]):
            white_layers[last_pos + i] = 255 - white_layers[i]
        last_pos += color_layers.shape[0]

    if do_dilate > 0:
        for i in range(color_layers.shape[0]):
            white_layers[last_pos + i] = dilate(white_layers[i],
                                                times=do_dilate)

    segmented_layers = [text_image_segmentation(white_layers[i],
                                                min_area=min_area,
                                                max_area=max_area)
                        for i in range(white_layers.shape[0])]

    # print([ra.shape for ra, sa in segmented_masks])
    # print([sa.shape for ra, sa in segmented_masks])
    imgs = np.vstack(tuple([ra for ra, sa in segmented_layers]))
    stats = np.concatenate(tuple([sa for ra, sa in segmented_layers]))
    return imgs, stats

def generate_component_map(stats, PROXIMITY_BOUND=3):

        """Avians Component Chains

Suppose we have $n$ components on an image, each having $k$ labels, for $p_i$ is
the probability of $k_i=l$

We will construct component chains that will list these components. C_1 and
C_2 will be two ends of these links and it will also include the distance
and direction information. We will set a maximum distance and consider only
those which are closer than this. 

Distance and direction information will make searches much easier. Each
component will have a key (id or index) and we will be able to list those items
close to a component easily, without searching the whole. It will also make
easier to convert these to OCR. 

The field of the table is this: 

| C1 | C2 | Direction | Distance |


Direction is the angle between centers of the components. For a more or less
horizontal writing, these should be close to 180 or 0. For the diacritic
components below or above the stem components, these should be between 45
and 135. 

The distance is measured in terms of the size of C1. If C1 has a width of 10
pixels and a height of 5 pixels, we will measure the distance in terms of
10pixels and a component C2 30 pixels away from C1 will have 3 in its distance
field. 

We have cx and cy elements for each component. From these we will make a table

| C   | C_0 | C_1 | C_2 | C_3 | C_4 |
| C_0 |  0  |     |     |     |     |
| C_1 |     | 0   |     |     |     |
| C_2 |     |     | 0   |     |     |
| C_3 |     |     |     | 0   |     |
| C_4 |     |     |     |     | 0   |

that shows the *raw* distance and angle. From this we can calculate the
direction and relative distance. 
        """
        
        n_comp = stats.shape[0]
        distance_table = np.zeros(shape=(n_comp, n_comp),
                                  dtype=np.float)
        angle_table = np.zeros(shape=(n_comp, n_comp),
                               dtype=np.float)

        relative_distance_table = np.zeros_like(distance_table)

        p = np.vstack((stats['cx'], stats['cy']))
        LOG.debug("=== p.shape ===")
        LOG.debug(p.shape)

        for i in range(n_comp): 
            for j in range(i): 
                p_i = p[:, i]
                p_j = p[:, j]
                distance_table[i, j] = nl.norm(p_i - p_j)
                distance_table[j, i] = distance_table[i, j]
                angle_table[i, j] = np.arctan2(p_i[1] - p_j[1], p_i[0] - p_j[0])
                angle_table[j, i] = np.arctan2(p_j[1] - p_i[1], p_j[0] - p_i[0])

        c_max = np.max(np.vstack((stats['w'], stats['h'])), axis=0)
        for i in range(n_comp):
            relative_distance_table[i] = distance_table[i] / c_max[i]

        pr = np.where(relative_distance_table < PROXIMITY_BOUND)
        component_map = np.zeros(dtype=[('c1', pr[0].dtype),
                                        ('c2', pr[1].dtype),
                                        ('distance', relative_distance_table.dtype),
                                        ('angle', angle_table.dtype)],
                                 shape=(pr[0].shape[0], ))
        component_map['c1'] = pr[0]
        component_map['c2'] = pr[1]
        component_map['distance'] = relative_distance_table[pr]
        component_map['angle'] = angle_table[pr]

        return component_map

def component_map_to_nx(stats, cmap): 
    g = nx.Graph()
    for i in range(stats.shape[0]): 
        s = stats[i]
        g.add_node(i, x=s['x'], y=s['y'], h=s['h'], w=s['w'], cx=s['cx'],
                   cy=s['cy'], area=s['area'])

    for i in range(cmap.shape[0]):
        cm = cmap[i]
        g.add_edge(cm['c1'], cm['c2'], distance=cm['distance'],
                   angle=cm['angle'])

    return g

def draw_rect_around_nonzero(img, padding=0, thickness=1, color=(255, 0, 255)):
    assert len(img.shape) == 2
    nz = img.nonzero()
    min0 = max(nz[0].min() - padding, 0)
    max0 = min(nz[0].max() + padding, img.shape[0])
    min1 = max(nz[1].min() - padding, 0)
    max1 = min(nz[1].max() + padding, img.shape[1])
    out_img = np.dstack((img, img, img))
    cv2.rectangle(out_img, (min1, min0), (max1, max0), thickness=thickness, color=color)
    return out_img

def yank_nonzero(img, padding=0): 
    nz = img.nonzero()
    min0 = max(nz[0].min() - padding, 0)
    max0 = min(nz[0].max() + padding, img.shape[0])
    min1 = max(nz[1].min() - padding, 0)
    max1 = min(nz[1].max() + padding, img.shape[1])
    return img[min0:max0, min1:max1].copy()

def img_bounding_box(img):
    "Finds nonzero elements' boundaries"
    nz = img.nonzero()
    res_arr = np.zeros(shape=(len(nz), 2), dtype=img.dtype)
    for i, nz_ch in enumerate(nz):
        if len(nz_ch) > 0:
            res_arr[i, 0] = nz_ch.min()
            res_arr[i, 1] = nz_ch.max() + 1  # To compan
    return res_arr


def component_boundaries(component_arr):
    bounding_box_arr = np.zeros(shape=(component_arr.shape[0],),
                                dtype=[('x', np.int32),
                                       ('y', np.int32),
                                       ('w', np.int32),
                                       ('h', np.int32)])
    n, orig_x, orig_y = component_arr.shape
    for i in range(n):
        bb = img_bounding_box(component_arr[i])
        bounding_box_arr[i]['x'] = bb[0, 0]
        bounding_box_arr[i]['y'] = bb[1, 0]
        w = bb[0, 1] - bb[0, 0]
        bounding_box_arr[i]['w'] = w
        h = bb[1, 1] - bb[1, 0]
        bounding_box_arr[i]['h'] = h
    return bounding_box_arr


def resize_boundaries(bounding_box_arr,
                      horizontal_ratio,
                      vertical_ratio):
    assert horizontal_ratio > 0
    assert vertical_ratio > 0
    # resize horizontal

    new_w = bounding_box_arr['w'] * horizontal_ratio
    new_w[new_w < 0] = 0
    w_diff = new_w - bounding_box_arr['w']
    new_x = bounding_box_arr['x'] - w_diff / 2
    new_x[new_x < 0] = 0

    # resize vertical
    new_h = bounding_box_arr['h'] * vertical_ratio
    new_h[new_h < 0] = 0
    h_diff = new_h - bounding_box_arr['h']
    new_y = bounding_box_arr['y'] - h_diff / 2
    new_y[new_y < 0] = 0

    new_bb_arr = np.empty_like(bounding_box_arr)
    new_bb_arr['x'] = np.round(new_x)
    new_bb_arr['y'] = np.round(new_y)
    new_bb_arr['w'] = np.round(new_w)
    new_bb_arr['h'] = np.round(new_h)

    return new_bb_arr


def decompose_img(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    component_arr = text_image_segmentation(img)
    return component_arr

# def img_colors(img):
#     if len(img.shape) == 3:
#         w, h, _ = img.shape
#         combined = np.dstack((img, np.zeros((w, h), np.uint8)))
#         uniq, uniq_indices = np.unique(combined.view(np.uint32), return_index=True)
#         return uniq_indices
#     elif len(img.shape) == 2:
#         return np.unique(img)
#     else:
#         raise Exception("Image should have 1 or 3 channels")


def colors_of(img):
    if len(img.shape) == 2:
        return np.unique(img)
    elif len(img.shape) == 3:
        color_set = set()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                color_set.add(tuple(img[i, j]))
        return np.vstack(color_set)
    else:
        raise Exception("Image should have 1 or 3 channels")


def color_masks(img):
    colors = colors_of(img)
    n_colors = colors.shape[0]
    color_mask_stack = np.zeros(shape=(n_colors, img.shape[0], img.shape[1]),
                                dtype=np.bool)
    for i, c in enumerate(colors):
        print(c.shape)
        print(img.shape)
        mask_0 = img[:, :, 0] == c[0]
        mask_1 = img[:, :, 1] == c[1]
        mask_2 = img[:, :, 2] == c[2]
        print(color_mask_stack[i].shape)
        color_mask_stack[i] = np.bitwise_and(mask_0, mask_1, mask_2)
    return color_mask_stack


def resize_components(component_arr, bounding_box_arr, size_x=64, size_y=64):
    n, orig_x, orig_y = component_arr.shape
    resized_arr = np.zeros(shape=(n,
                                  size_x,
                                  size_y), dtype=component_arr.dtype)
    LOG.debug("=== bounding_box_arr ===")
    LOG.debug(bounding_box_arr)

    LOG.debug("=== bounding_box_arr.shape ===")
    LOG.debug(bounding_box_arr.shape)

    LOG.debug("=== type(bounding_box_arr) ===")
    LOG.debug(type(bounding_box_arr))

    LOG.debug("=== n ===")
    LOG.debug(n)

    for i in range(n):
        LOG.debug("=== i ===")
        LOG.debug(i)
        bb = bounding_box_arr[i]
        LOG.debug("=== bb ===")
        LOG.debug(bb)
        x1 = bb['x']
        x2 = bb['x'] + bb['w']
        y1 = bb['y']
        y2 = bb['y'] + bb['h']
        if bb['area'] > 0:
            # BEWARE: x,y above are in OpenCV order, below we use NumPy order
            cpy = component_arr[i, x1:x2, y1:y2].copy()
            LOG.debug("=== (size_x, size_y, x1, x2, y1, y2) ===")
            LOG.debug((size_x, size_y, x1, x2, y1, y2))
            try:
                resized_arr[i] = cv2.resize(cpy,
                                            dsize=(size_x, size_y),
                                            interpolation=cv2.INTER_AREA)
            except cv2.error as e:
                LOG.warning("Resize Caused OpenCV error: {}".format(e))

    return resized_arr


def decompose_and_resize(img, size_x, size_y):
    decomposed, stats = decompose_img(img)
    return resize_components(decomposed, stats, size_x, size_y)


def create_word_image(text, fontname, filename, foreground, background, dpi=600):
    """creates a word image for the given text and file name"""
    # noqa
    bg = "#" + hex_color(background)
    fg = "#" + hex_color(foreground)
    cmd = 'pango-view --font="{font}" --text "{word}" --no-display --dpi="{dpi}" --background="{bg}" --foreground="{fg}" --output "{filename}"'
    cmd = cmd.format(font=fontname, word=text,
                     filename=filename, dpi=dpi, bg=bg, fg=fg)
    os.system(cmd)
    return filename


def get_gray_word_image(text, fontname, dpi, background, foreground, keep_image=True):
    # TODO: Add temp file name modifiable
    filename = "/dev/shm/" + str(uuid.uuid4()) + ".png"
    create_word_image(text, fontname, filename,
                      foreground=(foreground, foreground, foreground),
                      background=(background, background, background),
                      dpi=dpi)
    img = cv2.imread(filename, 0)
    if not keep_image:
        os.remove(filename)
    return img


def get_color_word_image(text, fontname, dpi, background, foreground):
    filename = "/tmp/" + str(uuid.uuid4()) + ".png"
    create_word_image(text, fontname, filename, foreground, background, dpi)
    img = cv2.imread(filename)
    os.remove(filename)
    return img


def get_white_on_black_word_images(text, fonts, dpis):
    imgs = []
    for dpi in dpis:
        for font in fonts:
            imgs.append(get_gray_word_image(text, fontname=font,
                                            dpi=dpi, background=0, foreground=255))
    return imgs


def write_image_array(arr, key, outdir="/tmp"):
    n_img = arr.shape[0]
    for i in range(n_img):
        cv2.imwrite("{}/arr-{}-{}.png".format(outdir, key, i), arr[i])


# def text_on_image(text_to_print,
#                   font_path="/usr/share/fonts/truetype/droid/DroidSansArabic.ttf",
#                   bgcolor=(0, 0, 0),
#                   fgcolor=(255, 255, 255),
#                   font_size=30,
#                   image_size=None,
#                   text_xy=None):

#     text_font = ImageFont.truetype(font_path, font_size)
#     if image_size is None:
#         # Get the size of image that can contain the text
#         size_img = Image.new("RGB", (1, 1), bgcolor)
#         size_draw = ImageDraw.Draw(size_img)
#         text_size = size_draw.textsize(text=text_to_print, font=text_font)
#         # extend a bit
#         text_size = (text_size[0] + 200, text_size[1] + 200)
#     else:
#         text_size = image_size

#     if text_xy is None:
#         text_xy = (100, 100)

#     text_img = Image.new("RGB", text_size, bgcolor)
#     text_draw = ImageDraw.Draw(text_img)
#     text_draw.text(xy=text_xy, text=text_to_print, font=text_font, fill=fgcolor)
#     return np.array(text_img)


# def arabic_on_image(text_to_print,
#                     font_path="/usr/share/fonts/truetype/droid/DroidSansArabic.ttf",
#                     bgcolor=(0, 0, 0),
#                     fgcolor=(255, 255, 255),
#                     font_size=30,
#                     image_size=None,
#                     text_xy=None):

#     # Convert the text to suitable ordering for image
#     reshaped_text = ar.reshape(text_to_print)
#     bidi_text = ba.get_display(reshaped_text)
#     return text_on_image(bidi_text,
#                          font_path,
#                          bgcolor,
#                          fgcolor,
#                          font_size,
#                          image_size,
#                          text_xy)


def draw_regions_on_image(img, regions, color=(0, 255, 255)):
    cv2.drawContours(img, [p.reshape(-1, 1, 2) for p in regions], -1, color)
    return img


def merge_image_array(img_arr, merge_value=0):
    "Combine all images into a single array"
    n_img, x, y = img_arr.shape
    merged = np.zeros(shape=(x, y), dtype=np.uint8)
    for i in range(n_img):
        merged += img_arr[i]
        if merge_value > 0:
            merged[merged.nonzero()] = merge_value
    return merged


def split_image(img_tuple, size_x, size_y, strides_x, strides_y):
    LOG.debug("=== img_tuple ===")
    LOG.debug(img_tuple)
    assert type(img_tuple) == type((1, 2)), "img_tuple should be a tuple"
    
    img = img_tuple[0]
    LOG.debug("=== img.shape ===")
    LOG.debug(img.shape)
    max_x, max_y = (img.shape[0], img.shape[1])
    x_coords = list(range(0, max_x, strides_x))
    y_coords = list(range(0, max_y, strides_y))
    n_outarray = len(x_coords) * len(y_coords)
    LOG.debug("=== np.ndim(img) ===")
    LOG.debug(np.ndim(img))
    LOG.debug("=== img.ndim ===")
    LOG.debug(img.ndim)
    n_ch = 1 if img.ndim == 2 else img.shape[2]
    img = np.atleast_3d(img)
    out_array = np.zeros(shape=(n_outarray, size_x, size_y, n_ch),
                         dtype=img.dtype)
    LOG.debug("=== img.shape ===")
    LOG.debug(img.shape)
    for i_out, coords in enumerate(itertools.product(x_coords,
                                                     y_coords)):
        x1 = coords[0]
        y1 = coords[1]
        x2 = x1 + size_x
        y2 = y1 + size_y
        # to avoid broadcast errors
        tmp = img[x1:x2, y1:y2]
        out_array[i_out, :tmp.shape[0], :tmp.shape[1]] = tmp
    return out_array


def split_image_list(images,
                     size_x,
                     size_y,
                     strides_x,
                     strides_y):
    LOG.debug("=== images ===")
    LOG.debug(images)
    array_list = [split_image(img,
                              size_x,
                              size_y,
                              strides_x,
                              strides_y) for img in images]
    return np.vstack(array_list)


def split_image_array(image_array,
                      size_x,
                      size_y,
                      strides_x,
                      strides_y):
    array_list = [split_image(image_array[i],
                              size_x,
                              size_y,
                              strides_x,
                              strides_y) for i in range(image_array.shape[0])]
    return np.vstack(array_list)


def resize_image_list(img_list,
                      size_list):
    out_list = []
    for img in img_list:
        for size_x, size_y in size_list:
            out_list.append(cv2.resize(img,
                                       dsize=(size_x, size_y),
                                       interpolation=cv2.INTER_AREA))
    return out_list
