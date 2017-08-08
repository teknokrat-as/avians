
import avians.util.image as aui
import avians.util.arabic as aua

import avians.nn.models.component_ctc_v6 as anmccv6
import avians.detect.nm_detect as adnd

import numpy as np
import cv2

import os

import argparse

import logging
LOG = logging.getLogger(__name__)

# TODO: Write the search model file here
SEARCH_MODEL_FILE = os.path.expandvars("$HOME/Annex/Arabic/ctc-v5-2017-01-13/weights.h5")
SEARCH_MODEL = anmccv6.ComponentModelCTC(SEARCH_MODEL_FILE)
TEXT_DETECTION_FUNC = adnd.find_text_components


def _array_x2(a):
    z = np.zeros(shape=a.shape,
                 dtype=a.dtype)
    return np.concatenate((a, z))


def load_labelmap(lm_file):
    f = open(lm_file, 'rb')
    contents = np.load(f)
    return contents

def _merge_and_sort_lm_results(lm_results, num_results=100):
    """lm_results is a dict[lm_file] = [(label, total_p, avg_p, cond_p, coords)]. we merge
    these into [lm_file, score, rx, ry, rw, rh] and sort by score and return
    the most likely.

    """
    LOG.debug("=== lm_results ===")
    LOG.debug(lm_results)

    LOG.debug("=== lm_results.values() ===")
    LOG.debug(lm_results.values())
    n_total_res = sum(
        [res.shape[0]
         for res in lm_results.values()
         if res is not None])
    longest_fn = max([len(fn) for fn in lm_results.keys()])

    merged_results = np.empty(dtype=[("filename", "S" + str(longest_fn)),
                                     ("score", np.float),
                                     ("avg_p", np.float),
                                     ("cond_p", np.float),
                                     ("total_p", np.float),
                                     ("x", np.int),
                                     ("y", np.int),
                                     ("w", np.int),
                                     ("h", np.int)],
                              shape=(n_total_res,))
    mb = 0
    for fn, fn_results in lm_results.items():
        if fn_results is not None:
            me = mb + fn_results.shape[0]
            merged_results[mb:me]['filename'] = bytes(fn, 'utf-8')
            merged_results[mb:me]['score'] = fn_results['conditional_probability']
            merged_results[mb:me]['cond_p'] = fn_results['conditional_probability']
            merged_results[mb:me]['total_p'] = fn_results['total_probability']
            merged_results[mb:me]['avg_p'] = fn_results['average_probability']
            merged_results[mb:me]['x'] = fn_results['min_x']
            merged_results[mb:me]['y'] = fn_results['min_y']
            merged_results[mb:me]['w'] = fn_results['max_x'] - fn_results['min_x']
            merged_results[mb:me]['h'] = fn_results['max_y'] - fn_results['min_y']
    sort_index = np.argsort(merged_results['score'])
    return merged_results[sort_index][:num_results]

def search_string_in_dir(directory,
                         visenc_string,
                         num_results=100):
    labelmap_files = aui.recursive_file_list(directory,
                                             f_regex=r'.*\.lm2$')
    assert len(labelmap_files) > 0

    lm_results = {}
    for lmf in labelmap_files:
        lm = load_labelmap(lmf)
        lm_results[lmf] = search_string_in_component_map(lm,
                                                         visenc_string)
    final_results = _merge_and_sort_lm_results(lm_results)
    return final_results[:num_results]

# def search_string_in_labelmap(lm,
#                               visenc_string):
#     "Searches the string encoded in visenc in a labelmap structure"
#     # strip visenc from dots
#     stems = aua.delete_visenc_diacritics(visenc_string)
#     # split visenc
#     split_stems = aua.split_visenc(stems)
#     # search units in labels
#     label_indices = [_find_label_indices(lm, s) for s in split_stems]
#     # merge units by sensibility:
#     # the result (rotated) rectangle should be smaller than the factor * sum
#     # of their individual rectangles
#     result_stats, result_labels = _label_rectangles(lm, label_indices)
#     # result is a rectangle on an image, containing labels and sum of scores
#     return result_stats, result_labels

def neighbors_with_label(stem_label, lm, label_index, deg_min=0,
                         deg_max=360, among_closest=10):
    cmap = lm['component_map']
    labels = lm['labels']
    rad_min = np.deg2rad(deg_min)
    rad_max = np.deg2rad(deg_max)
    map_index = np.bitwise_and(cmap['c1'] == label_index,
                               np.bitwise_and(cmap['angle'] > rad_min,
                                              cmap['angle'] < rad_max))
    c1_map = cmap[map_index]
    if c1_map.shape[0] > among_closest:
        c1_map_argsort_dist = c1_map['distance'].argsort()
        c1_map = c1_map[c1_map_argsort_dist[:10]]

    LOG.debug("=== c1_map ===")
    LOG.debug(c1_map)
    LOG.debug("=== labels.shape ===")
    LOG.debug(labels.shape)

    c1_neighbor_labels = labels[c1_map['c2'], :]
    LOG.debug("=== c1_neighbor_labels.shape ===")
    LOG.debug(c1_neighbor_labels.shape)

    LOG.debug("=== c1_neighbor_labels ===")
    LOG.debug(c1_neighbor_labels)
    res = np.where(c1_neighbor_labels['label'] == bytes(stem_label, 'ascii'))
    LOG.debug("=== res ===")
    LOG.debug(res)
    return res

def _search_stems(lm, stem_list):
    LOG.debug("=== _search_stems ===")
    LOG.debug("=== lm ===")
    LOG.debug(lm)
    LOG.debug("=== stem_list ===")
    LOG.debug(stem_list)
    label_indices = _find_label_indices(lm, stem_list[0])
    LOG.debug("=== label_indices ===")
    LOG.debug(label_indices)

    MAX_RESULTS = 1000
    # We store indices in result table
    result_table = np.zeros(dtype=[("label_i", np.int), ("label_j", np.int)],
                            shape=(MAX_RESULTS, len(stem_list)))
    LOG.debug("=== result_table.shape ===")
    LOG.debug(result_table.shape)
    LOG.debug("=== result_table[:len(label_indices, 0].shape ===")
    LOG.debug((result_table[:len(label_indices), 0]).shape)
    LOG.debug("=== label_indices ===")
    LOG.debug(label_indices)
    result_table[:len(label_indices[0]), 0]['label_i'] = label_indices[0]
    result_table[:len(label_indices[1]), 0]['label_j'] = label_indices[1]
    stem_i = 1
    searching_in = list(range(len(label_indices)))
    available_indices = [i for i in range(MAX_RESULTS) if i not in searching_in]
    while stem_i < len(stem_list) and len(searching_in) > 0:
        LOG.debug("stem_i: {}\n".format(stem_i))
        for si, label_index in enumerate(searching_in):
            neighbors = neighbors_with_label(stem_list[stem_i],
                                             lm, label_index,
                                             deg_min = 0,
                                             deg_max = 360,
                                             among_closest=20)
            LOG.debug("len(neighbors[0]): {}\n".format(len(neighbors[0])))
            if len(neighbors[0]) > 0:
                for ni in range(neighbors[0].shape[0]):
                    if ni >= 1:
                        # We have branching, current label has more than 1
                        # neighbor with the necessary label if we have more
                        # than 1 element, we use available indices
                        if len(available_indices) == 0:
                            current_max_index = result_table.shape[0]
                            result_table = _array_x2(result_table)
                            available_indices = list(range(
                                current_max_index,
                                result_table.shape[0]))
                            LOG.debug("=== INCREASED result_table.shape ===")
                            LOG.debug(result_table.shape)

                        ti = available_indices[0]
                        del available_indices[0]
                        # Copy previous result to the current one
                        result_table[ti, :(stem_i-1)] = result_table[
                            label_index, :(stem_i-1)]
                        searching_in.append(ti)
                    else:
                        ti = label_index
                    result_table[ti, stem_i] = (neighbors[0][ni],
                                                neighbors[1][ni])
            else: # if we have no neighbors
                # Remove from the search pool
                del searching_in[si]
                available_indices.append(label_index)
        stem_i += 1

    LOG.debug("final searching_in: {}".format(searching_in))
    res = result_table[searching_in, :]
    LOG.debug("res: {}".format(res))
    return res


def _build_results(lm, indices):
    found_labels = np.zeros(dtype=[("string", "S64"),
                                   ("total_probability", np.float),
                                   ("average_probability", np.float),
                                   ("conditional_probability", np.float),
                                   ("min_x", np.int),
                                   ("min_y", np.int),
                                   ("max_x", np.int),
                                   ("max_y", np.int)],
                            shape=(indices.shape[0],))

    labels = lm['labels']
    stats = lm['stats']
    LOG.debug("indices: {}".format(indices))
    for ind_i in range(indices.shape[0]):
        current_str = ""
        total_prob = 0
        conditional_prob = 1
        min_x = float("inf")
        max_x = 0
        min_y = float("inf")
        max_y = 0
        for ind_j in range(indices.shape[1]):
            label_i = indices['label_i'][ind_i, ind_j]
            label_j = indices['label_j'][ind_i, ind_j]
            LOG.debug("=== (label_i, label_j) ===")
            LOG.debug((label_i, label_j))
            current_str += labels[label_i, label_j]['label'].decode('ascii')
            LOG.debug("=== current_str ===")
            LOG.debug(current_str)
            total_prob += labels[label_i, label_j]['prob']
            LOG.debug("=== total_prob ===")
            LOG.debug(total_prob)
            conditional_prob *= labels[label_i, label_j]['prob']
            LOG.debug("=== conditional_prob ===")
            LOG.debug(conditional_prob)
            x = stats[label_i]['x']
            y = stats[label_i]['y']
            w = stats[label_i]['w']
            h = stats[label_i]['h']
            min_y = y if y < min_y else min_y
            min_x = x if x < min_x else min_x
            max_x = (x+w) if (x+w) > max_x else max_x
            max_y = (y+h) if (y+h) > max_y else max_y
        found_labels[ind_i] = (current_str,
                               total_prob,
                               total_prob / indices.shape[1],
                               conditional_prob,
                               min_x,
                               min_y,
                               max_x, max_y)
    return found_labels

def search_string_in_component_map(lm,
                                   visenc_string):
    """Searches the string via component map by starting from initial letter and
moving by proximity and direction

    """
    LOG.debug("Searching {}".format(visenc_string))
    # strip visenc from dots
    stems = aua.delete_visenc_diacritics(visenc_string)
    LOG.debug(stems)
    # split visenc
    split_stems = aua.split_visenc(stems)
    LOG.debug(split_stems)
    indices = _search_stems(lm, split_stems)
    LOG.debug("indices: {}".format(indices))
    res = _build_results(lm, indices)
    LOG.debug(res)
    return res

def _find_label_indices(labelmap, l):
    LOG.debug("=== find_label_indices(labelmap, l) ===")
    LOG.debug((labelmap, l))
    labels = labelmap['labels']
    LOG.debug("=== labels ===")
    LOG.debug(labels)
    LOG.debug("=== type(labels) ===")
    LOG.debug(type(labels))
    LOG.debug("=== labels.shape ===")
    LOG.debug(labels.shape)
    LOG.debug("=== labels['label'] ===")
    LOG.debug(labels['label'])
    if labels.shape == (0,):
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.int64))
    else:
        cond = labels['label'] == bytes(l, 'utf-8')
        LOG.debug("=== cond ===")
        LOG.debug(cond)
        indices = np.where(cond)
    LOG.debug("indices: {}\n".format(indices))
    return indices


def _label_rectangles(labelmap, label_index_list, filter_factor=1.5):
    """This function filters groups of labels by position and size.

    (1) Labels should be sorted right to left, that is Arabic writing order.
    (2) The rectangle of merged labels should be smaller than the sum of
    individual rectangles of labels

    The labels are checked in the order they are found in label_index_list.  Two
    consecutive label candidates are merged if only they meet these criteria.

    """

    labels = labelmap['labels']
    stats = labelmap['stats']

    # turn the list into a stack
    label_index_list.reverse()
    LOG.debug("=== li ===")
    LOG.debug(label_index_list[0])
    LOG.debug("=== li[0] ===")
    LOG.debug(label_index_list[0][0])
    LOG.debug("=== li[0].shape ===")
    LOG.debug(label_index_list[0][0].shape)
    LOG.debug("=== stats.shape ===")
    LOG.debug(stats.shape)
    LOG.debug("=== labels.shape ===")
    LOG.debug(labels.shape)

    # TODO: It may be wiser to drop everything if label_index_list contains
    # empty indices, that is shapes not found on the document. But for smaller
    # shapes, we opted to include all the found elements. We can also add a 0
    # probability special element to reduce the overall probability
    stats_list = [stats[li[0]]
                  for li in label_index_list if li[0].shape != (0,)]
    labels_list = [labels[li]
                   for li in label_index_list if li[0].shape != (0,)]
    assert len(stats_list) == len(labels_list)

    while len(stats_list) > 1:
        stats_1 = stats_list.pop()
        labels_1 = labels_list.pop()
        stats_2 = stats_list.pop()
        labels_2 = labels_list.pop()

        merged_stats, merged_labels = _merge_two_labelsets(stats_1, labels_1,
                                                           stats_2, labels_2)

        stats_list.append(merged_stats)
        labels_list.append(merged_labels)

    if len(stats_list) > 0 and stats_list[0].shape[0] > 0:
        return (stats_list[0], labels_list[0])
    else:
        return (None, None)

def _merge_two_labelsets(stats_1, labels_1, stats_2, labels_2):
    "Merge two sets according to their positions"
    res_size = stats_1.shape[0] * stats_2.shape[0]
    res_stats = np.empty(shape=(res_size,),
                         dtype=stats_1.dtype)
    res_labels = np.empty(shape=(res_size,),
                          dtype=labels_1.dtype)
    filled_result_counter = 0
    for i in range(stats_1.shape[0]):
        for j in range(stats_2.shape[0]):
            # TODO: Check the following conditions
            s1 = stats_1[i]
            s2 = stats_2[j]
            l1 = labels_1[i]
            l2 = labels_2[j]
            right_to_left = s1['x'] > s2['x']
            width = abs(s1['cx'] - s2['cx']) < (s1['w'] + s2['w'])
            if right_to_left and width:
                rs = res_stats[filled_result_counter]
                rl = res_labels[filled_result_counter]
                rs['x'] = s2['x']
                rs['y'] = min(s1['y'], s2['y'])
                rs['w'] = min(s1['x'] + s1['w'], s2['x'] + s2['w']) - rs['x']
                rs['h'] = min(s1['y'] + s1['h'], s2['y'] + s2['y']) - rs['y']
                rs['cx'] = rs['x'] + rs['w'] / 2
                rs['cy'] = rs['y'] + rs['h'] / 2
                rs['area'] = s1['area'] + s2['area']
                rl['label'] = l1['label'] + l2['label']
                rl['prob'] = (l1['prob'] + l2['prob']) / 2

                filled_result_counter += 1
    return (res_stats[:filled_result_counter],
            res_labels[:filled_result_counter])

def labelmap_to_dir(lm2_file, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    lm = load_labelmap(lm2_file)
    imgs = lm['images']
    pad_length = len(str(imgs.shape[0]))
    for i in range(imgs.shape[0]):
        whole_img_f = os.path.join(output_dir, "whole-{num:0={pad}d}.png".format(num=i, pad=pad_length))
        img_f = os.path.join(output_dir, "img-{num:0={pad}d}.png".format(num=i, pad=pad_length))
        img = imgs[i]
        whole_img = aui.draw_rect_around_nonzero(img, padding=7, thickness=2)
        comp_img = aui.yank_nonzero(img, padding=3)
        cv2.imwrite(whole_img_f, whole_img)
        cv2.imwrite(img_f, comp_img)

    labels = np.zeros(dtype='U70', shape=lm['labels'].shape)
    stats = np.zeros(dtype='U70', shape=lm['stats'].shape)
    cmap = np.zeros(dtype='U70', shape=lm['component_map'].shape)
    lbl = lm['labels']
    for i in range(lbl.shape[0]):
        labels[i] = tuple(["{} : {:.2f}".format(
            lbl['label'][i, j].decode('utf-8'),
            lbl['prob'][i, j]) for j in range(lbl.shape[1])])
        s = lm['stats'][i]
        stats[i] = "{},{},{},{},{},{},{}".format(s['x'], s['y'], s['w'], s['h'], s['cx'], s['cy'], s['area'])

    np.savetxt(os.path.join(output_dir, "labels.txt"),
               labels, fmt="%s",
               delimiter=',')
    np.savetxt(os.path.join(output_dir, "stats.txt"),
               lm['stats'], fmt='%s', delimiter=',')
    np.savetxt(os.path.join(output_dir, "cmap.txt"),
               lm['component_map'], fmt=('%d', '%d', '%.2f', '%.2f'), delimiter=',')


def image_to_labelmap(image_file,
                      write_output_to='auto'):
    """For each image in image_lib_dir:
    (a) load image
    (b) detect text
    (c) perform text segmentation
    (d) classify each component
    (e) merge the data
    (f) save the data similar to image
    """
    img = cv2.imread(image_file)
    text_objects = TEXT_DETECTION_FUNC(img)
    images, stats = text_objects
    if text_objects[0] is not None:
        labels = SEARCH_MODEL.classify_text_objects(text_objects)
        component_map = aui.generate_component_map(stats)
    else:
        labels = np.array([])
        component_map = np.array([])

    if write_output_to:
        if write_output_to == 'auto':
            outfile = os.path.splitext(image_file)[0] + '.lm2'
        else:
            outfile = write_output_to
        fout = open(outfile, 'wb')
        np.savez_compressed(fout,
                            images=text_objects[0],
                            stats=text_objects[1],
                            labels=labels,
                            component_map=component_map)

    return outfile, text_objects[0], text_objects[1], labels, component_map

def isolate_largest_components(segmented_word_imgs):
    return [np.argmax(s[1]['area']) for s in segmented_word_imgs]


def train_custom_model_for_images(train_dir):
    model_obj = SEARCH_MODEL()
    model_obj.train_model(train_dir)
    return model_obj

def main_dispatch(word, image_dir, output_dir):
    assert False

def main():
    random_output_dir = "/tmp/avians-results-{}".format(np.random.randint())
    parser = argparse.ArgumentParser(description="""
    Searches a word in a set of image files
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('word',
                        help="The word to be sought in visenc")
    parser.add_argument('image-dir',
                        help="Directory that contains the image files")
    parser.add_argument('output-dir',
                        help="Directory to put the output images",
                        default=random_output_dir)
    args = vars(parser.parse_args())
    main_dispatch(**args)


if __name__ == '__main__':
    main()
