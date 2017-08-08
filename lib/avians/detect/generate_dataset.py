
import cv2
import numpy as np
import numpy.random as nr
import avians.util.image as aui
import argparse
import os
import uuid
from avians.util.arabic_defs import ARABIC_FONTS
import multiprocessing as mp
import logging
LOG = logging.getLogger(__name__)

def resize_single_image(input_file, sizex, sizey):
    img = cv2.imread(input_file)
    new_img = cv2.resize(img, (sizex, sizey))
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    return new_img

def resize_32x32(input_file): 
    return resize_single_image(input_file, 32, 32)

parallel_processable_sizes = {(32, 32): resize_32x32}

def generate_grayscale_1D(input_dir, sizex, sizey, class_value):
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    if (sizex, sizey) in parallel_processable_sizes: 
        pool = mp.Pool(mp.cpu_count() - 1)
        func = parallel_processable_sizes[(sizex, sizey)]
        resized_images = pool.map(func, image_files)
    else: 
        resized_images = [resize_single_image(f, sizex, sizey) for f in image_files]
    
    rows = len(resized_images)
    columns = sizex * sizey + 1
    dataset = np.zeros(shape=(rows, columns), dtype=np.uint8)
    class_value = int(class_value)
    for i, img in enumerate(resized_images): 
        dataset[i, 0] = class_value
        dataset[i, 1:] = img.flatten()
    return dataset

def grayscale_1D(args): 
    """Resizes the image file to (sizex, sizey) having given channels and outputs to the output_dir
    """
    # input_dir, sizex, sizey, channels, output_dir

    dataset = generate_grayscale_1D(args.input_dir, args.sizex, args.sizey, args.class_value)
    np.save(dataset, args.output_fiholes)

DISTANCE_TRANSFORM_METRIC = cv2.DIST_L1
DISTANCE_TRANSFORM_BLOCKSIZE = 3
TEXT_CONTOUR_MINIMUM_SIZE = 16

def generate_random_arabic_text_image(lines):
    num_words_in_text = nr.randint(1, 10)
    # LOG.debug("=== lines ===")
    # LOG.debug(lines)
    
    text = " ".join([w.strip() for w in list(nr.choice(lines, num_words_in_text))])
    fgcolor = (nr.randint(50, 255), nr.randint(50, 255), nr.randint(50, 255))
    bgcolor = (nr.randint(50, 255), nr.randint(50, 255), nr.randint(50, 255))
    font = nr.choice(ARABIC_FONTS)
    dpi = nr.choice(list(range(50, 1200, 50)))
    filename = "/tmp/text-image-{}.png".format(uuid.uuid4())
    aui.create_word_image(text, font, filename, fgcolor, bgcolor, dpi)
    return filename

def resized_contour_distance_transform(img, contour_structure, sizex, sizey): 
    contour, holes, level = contour_structure
    min_x = np.min(contour[:, 0, 0])
    min_y = np.min(contour[:, 0, 1])
    max_x = np.max(contour[:, 0, 0])
    max_y = np.max(contour[:, 0, 1])
    w = max_x - min_x + 1
    h = max_y - min_y + 1

    contour_img = np.zeros(shape=(h, w), dtype=np.uint8)
    _contour = np.hstack((contour[:, :, 0] - min_x, contour[:, :, 1] - min_y))
    _holes = [np.hstack((hole[:, :, 0] - min_x, hole[:, :, 1] - min_y)) for hole in holes]
    cv2.drawContours(contour_img, [_contour], -1, 255, cv2.FILLED)
    cv2.drawContours(contour_img, _holes, -1, 0, cv2.FILLED)

    distance_transform = cv2.distanceTransform(contour_img, DISTANCE_TRANSFORM_METRIC, DISTANCE_TRANSFORM_BLOCKSIZE)
    # LOG.debug("=== (w, h) ===")
    # LOG.debug((w, h))
    
    if max(w, h) > min(sizex, sizey): 
        return cv2.resize(distance_transform, (sizex, sizey))        
    else:
        distance_transform.resize((sizex, sizey))
        return distance_transform
    
def contour_region(img, contour): 
    LOG.debug("=== img.shape ===")
    LOG.debug(img.shape)
    
    mask = np.zeros_like(img)
    LOG.debug("=== mask.shape ===")
    LOG.debug(mask.shape)
    LOG.debug("=== contour ===")
    LOG.debug(contour)
    
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    masked = np.bitwise_and(mask, img)
    LOG.debug("=== masked ===")
    LOG.debug(masked)
    (y, x, h, w) = cv2.boundingRect(contour)
    LOG.debug("=== (x, y, w, h) ===")
    LOG.debug((x, y, w, h))
    if len(img.shape) == 2:
        return masked[x:(x+w), y:(y+h)]
    else: 
        return masked[x:(x+w), y:(y+h), :]

def resized_contour_region(img, contour, sizex, sizey): 
    # contour, holes, level = contour_structure
    contour_img = contour_region(img, contour)
    LOG.debug("=== img.shape ===")
    LOG.debug(img.shape)
    LOG.debug("=== contour_img.shape ===")
    LOG.debug(contour_img.shape)
    
    resized_contour_img = cv2.resize(contour_img, (sizex, sizey))
    return resized_contour_img

def resized_contour_regions(imgf, sizex, sizey):
    img = cv2.imread(imgf, 0)
    img, contours_and_holes = aui.enlarge_and_get_contours(img)
    contours = [cah[0] for cah in contours_and_holes]
    contour_regions = [resized_contour_region(img, c, sizex, sizey) for c in contours]
    return contour_regions

def resized_contour_regions_parallel(args): 
    return resized_contour_regions(args[0], args[1], args[2])

def generate_resized_grayscale_contours(input_dir, sizex, sizey, class_value): 
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    pool_list = [(imgf, sizex, sizey) for imgf in image_files]
    resized_images = [resized_contour_regions_parallel(a) for a in pool_list]
    # pool = mp.Pool(mp.cpu_count() - 1)
    # resized_images = pool.map(resized_contour_regions_parallel, pool_list, chunksize=5)

    rows = len(resized_images)
    columns = sizex * sizey + 1
    dataset = np.zeros(shape=(rows, columns), dtype=np.uint8)
    class_value = int(class_value)
    for i, img in enumerate(resized_images): 
        dataset[i, 0] = class_value
        dataset[i, 1:] = img.flatten()
    return dataset

def main(): 
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Subcommands to give the utility")

    grayscale_1D_p = subparsers.add_parser('grayscale-1D', aliases=['gs1'], help=grayscale_1D.__doc__)

    grayscale_1D_p.add_argument("input-dir",
                                help="Directory that contains image files")

    grayscale_1D_p.add_argument("output-file",
                                help="Numpy file to store final images")

    grayscale_1D_p.add_argument("--sizex",
                                help="Number of rows in final images",
                                default=32)

    grayscale_1D_p.add_argument("--sizey", 
                                help="Number of columns in final images",
                                default=32)

    grayscale_1D_p.add_argument("--class-value", 
                                help="Class value for these images",
                                default=0)

    grayscale_1D_p.set_defaults(func=grayscale_1D)
