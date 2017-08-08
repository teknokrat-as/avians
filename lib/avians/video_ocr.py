from avians.ocr import single_model_ocr
from avians.nn.models.component_ctc_v5 import ComponentModelCTC_v5 as model
from avians.util.arabic import visenc_to_arabic
import re
import os
import argparse
import cv2
import logging
LOG = logging.getLogger(__name__)

def lm_to_text(lm):
    line_groups = find_lines(lm)
    output = ""
    for lg in line_groups:
        s = sort_components(lg)
        line_text = line_to_text(lm)
        output += "{}\n".format(line_text)
    return output

def find_lines(lm):
    "Finds the lines of text on a map"
    stats = lm['stats']
    cmap = lm['component_map']

    # Find elements which have no connection to an element to the right
    angle_range = (pi/6, 11 * pi/6)
    has_right_el = np.where(cmap['angle'] > angle_range[0]
                              and cmap['angle'] < angle_range[1])
    comp_has_right_el = cmap['c1'][has_right_el]
    line_heads =  cmap['c1'] not in np.unique(comp_has_right_el)
    lines = []
    for l in line_heads:
        line_of_l = _find_line_of(l, lm)
        lines.append((l, line_of_l))

    lines = _sort_from_top_left_to_bottom_right(lines, lm)
    return lines


def _find_line_of(cid, lm):
    current_point = (stats[cid]['x'], stats[cid]['y'])
    neighbor_angle_range (5 * pi/6, 7 * pi/6)
    diacritic_size_bound = 0.1
    current_cid = cid
    elements = []
    while current_cid:
        # find all neighbors
        neighbors = np.where(cmap['c1'] == current_cid
        and cmap['angle'] < neighbor_angle_range[0]
        and cmap['angle'] > neighbor_angle_range[1])
        # discard diacritics
        neighbor_sizes = stats[neighbors['c1']]['area']
        min_neighbor_area = diacritic_size_bound * current_cid_area
        stem_neighbors = neighbor_sizes > min_neighbor_area
        min_distance_stem = argmin(stem_neighbors['distance'])
        elements.append(current_cid)
        if min_distance_stem:
            current_cid = min_distance_stem
        else:
            current_cid = None
    return elements

def _sort_from top_left_to_bottom_right(lines, lm):
    s = lm['stats']
    locations = [(s[head]['x'], s[head]['y'], head, els) for head, els in lines]

    locations = sorted(locations,
                       key = lambda e: e[1] * 100000 + e[0])
    return [(h, el) for x, y, h, el in locations]

def load_model(weights_file="$HOME/Repository/avians/image_ocr/Monday, 24. October 2016 12.53PM/weights03.h5"): 
    m = model(weights_file)
    return m

def shard_video(video_file, output_dir, frames_per_second=1):
    """
    # FIXME Loads and divides a video_file and writes results into output_dir
    """
    video = cv2.VideoCapture(video_file)

    LOG.debug("=== VIDEO READ ===")
    ret, frame = video.read()
    LOG.debug(ret)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    LOG.debug("=== fps ===")
    LOG.debug(fps)
    step = round((fps / frames_per_second) - 1) 
    step = max(1, step)
    LOG.debug("=== step ===")
    LOG.debug(step)
    
    frame_i = 0

    out_file_root = os.path.join(output_dir, os.path.splitext(os.path.basename(video_file))[0])

    while video:
        grabbed = video.grab()
        
        if (frame_i % step) == 0:
            success, img = video.retrieve()
            if success:
                cv2.imwrite(out_file_root + str((frame_i // step) + 1) + '.png', img)
            else:
                break
                    
        frame_i += 1
    
def detect_text_regions(image_file):
    """
    # TODO Runs the text detector on an image file and returns the textual
    rectangular regions

    """
    pass

def shard_image(image_file):
    """
    # TODO: Finds edges on the image and returns an array of images that contain one component per image
    """
    pass

def prepare_for_detection(component_array, img_w, img_h, items_per_component=3):
    """
    # TODO: Prints all items on img_w*img_h rectangles by resizing. It randomly changes the location of elements and returns items_per_component images for each component in component_array.
    """
    pass



def ocr_func(input_file):
    visenc_text = read_with_letter_group(input_file,
                                         os.path.expandvars("$HOME/Repository/harfbin/current-model.h5"),
                                         os.path.expandvars("$HOME/Repository/harfbin/current-dataset.npz"))

    print(visenc_text)

    arabic_text = visenc_to_arabic(visenc_text)

    return arabic_text

def main():

    parser = argparse.ArgumentParser(description="""
    Finds the labels of each component by checking their location
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input-file", help="File to put into OCR")
    parser.add_argument("--model",
                        help="CNN model to recognize the letter",
                        default=os.path.expandvars("$HOME/Datasets/leylamecnun/letter-group-model.h5"))

    parser.add_argument("--dataset",
                        help=".npz file that contains the num->class and class->num dicts, along with the training set",
                        default=os.path.expandvars("$HOME/Datasets/leylamecnun/letter-group-dataset.npz"))

    parser.add_argument("--output-labelmap",
                        help="Outputs the labelmap file",
                        default=None)

    args = vars(parser.parse_args())

    text = read_with_letter_group(args["input-file"],
                                  args["model"],
                                  args["dataset"],
                                  labelmap_file=args["output_labelmap"])

    print(text)

    return 0

if __name__ == "__main__":
    main()
