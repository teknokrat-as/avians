import avians.util.image as aui
import avians.util.arabic as aua
import avians.nn.models.component_autoencoder_v1 as anmcav1
import avians.detect.text_detection_v4 as adtdv4

import itertools
import numpy.linalg as nl
import numpy as np
import cv2

import os

import argparse

import logging
LOG = logging.getLogger(__name__)

DEFAULT_TEXT_DETECTION_MODEL = os.path.expandvars("$HOME/Annex/Arabic/text-detection-model-accuracy-0.99/text-detection-model-2016-06-15-11:52:27.h5")  # noqa

def search_word_in_library(word,
                           image_lib_dir,
                           feature_extractor,
                           fonts=None,
                           feature_size=64,
                           strides=32,
                           max_results=10,
                           output_dir="/tmp"):

    features_table, size_table, distance_table = calculate_word_features(word,
                                                                         feature_extractor,
                                                                         feature_size=feature_size)

    largest_component_indices = find_largest_component_indices(size_table)
    largest_component_features = features_table[:, 
                                                largest_word_component_indices, 
                                                :]
    min_relative_size = size_table.min()
    max_relative_distance = distance_table.max()

    # word_patches = _word_patches(word, feature_size, strides=strides)
    image_files = aui.recursive_file_list(image_lib_dir)
    image_file_dict = {fn: i for i, fn in enumerate(image_files)}
    images = [cv2.imread(img_f) for img_f in image_files]
    page_results = []
    for i, img in enumerate(images):
        res_img = search_in_image_via_components(features_table,
                                                     largest_component_indices,
                                                     min_relative_size,
                                                     max_relative_distance,
                                                     img,
                                                     feature_extractor,
                                                     feature_size=feature_size)
        page_results.append((image_files[i], res_img))

#    merged_results = merge_page_results(page_results)
    # sorted_results = sorted(page_results, key=lambda p: p[1].min())
    # result_imgs = {}
    # considered_results = sorted_results[:max_results]
    # min_score_to_consider = sorted_results[-1][1].min()
    # # max_score_to_consider = sorted_results['score'][considered_results.shape[0]] * 2
    # for img_f, r in considered_results:
    #     points = np.where(r <= min_score_to_consider)
    #     LOG.debug("=== points ===")
    #     LOG.debug(points)
    #     result_img = images[image_file_dict[img_f]]
    #     LOG.debug("=== result_img.shape ===")
    #     LOG.debug(result_img.shape)
    #     b_channel, g_channel, r_channel = cv2.split(result_img)
    #     alpha_channel = np.ones_like(b_channel) * 128
    #     LOG.debug("=== alpha_channel ===")
    #     LOG.debug(alpha_channel)
    #     LOG.debug("=== alpha_channel.shape ===")
    #     LOG.debug(alpha_channel.shape)
    #     alpha_channel[points] = 255
    #     result_img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    #     result_imgs[img_f] = result_img

        # res_imgf = image_files[r["file"]]
        # orig_img = images[image_file_dict[res_imgf]]
        # if res_imgf not in result_imgs:
        #     # result_imgs[res_imgf] = np.zeros(shape=orig_img.shape,
        #     #                                  dtype=np.float) # * max_score_to_consider
        #     result_imgs[res_imgf] = orig_img.copy()
        # res_img = result_imgs[res_imgf]
        # x1 = r["x"]
        # x2 = np.int32(r["x"] + r["w"] * 1.10)
        # y1 = r["y"]
        # y2 = np.int32(r["y"] + r["h"] * 1.10)
        # # cv2.imwrite("/tmp/res-{}.png".format(r['score']),
        # #             orig_img[x1:x2, y1:y2])
        # cv2.rectangle(res_img, (y1, x1), (y2, x2), (0, 0, 255), 2)
        # res_img[x1:x2, y1:y2] = orig_img[x1:x2, y1:y2]  # r["score"]

    output_imgs = {f: img for f, img in page_results}

    # output_imgs = {}
    # for rf in result_imgs:
    #     orig_img = images[image_file_dict[rf]]
    #     res_mask = max_score_to_consider - result_imgs[rf]
    #     LOG.debug("=== res_mask ===")
    #     LOG.debug(res_mask)
    #     res_img = np.zeros_like(orig_img)
    #     res_img[res_mask < max_score_to_consider] = orig_img[res_mask < max_score_to_consider]
    #     output_imgs[rf] = res_img

    for rf, out_img in output_imgs.items():
        out_fn = os.path.join(output_dir,
                              os.path.basename(rf) + ".png")
        cv2.imwrite(out_fn, out_img)
    return out_img

def filter_text_components(text_detection_model, component_arr):
    predictions = text_detection_model.predict(component_arr)
    return predictions > 0.01

def search_in_image_via_hook_algorithm(word_features,
                                       largest_component_indices,
                                       min_relative_size,
                                       max_relative_distance,
                                       img,
                                       feature_extractor,
                                       feature_size): 

    # word_feature_table = np.empty(shape=(len(word_components),
    #                                      max_components_per_word,
    #                                      feature_size),
    #                               dtype=np.float)

    components, stats = aui.image_segmentation(img,
                                               do_color_quantization=True)

    # page_features = array((components.shape[0], feature_size))

    page_features = feature_extractor.predict(components)
    page_component_cross_sizes = calculate_cross_sizes(stats)
    page_component_cross_distances = calculate_cross_distances(stats)
    seed_indices = find_seed_indices(largest_component_features,
                                     page_features)

    word_candidates = find_closest_smaller_elements(page_component_cross_sizes, 
                                                    page_component_cross_distances,
                                                    seed_indices,
                                                    min_relative_size,
                                                    max_relative_distance)
    
    word_scores = calculate_word_scores(seed_indices,
                                        word_candidates,
                                        page_features,
                                        features_table,
                                        stats)
    return word_scores

def calculate_word_scores(seed_indices,
                          word_candidates,
                          page_features,
                          word_features,
                          stats): 

    word_scores = np.empty(shape=word_candidates.shape,
                           dtype=np.float)

    for i, si in enumerate(seed_indices): 
        page_features_to_compare = page_features[word_candidates[i]]
        word_scores[i] = calculate_minimum_word_score(page_features_to_compare, 
                                                      word_features)
    return word_scores


def calculate_minimum_word_score(page_features,
                                 word_feature_table): 
    # page features: array(components x vector_size)
    # word_features: array(word x components x vector_size)
    
    calculation_table = np.empty(shape=(word_feature_table.shape[0],
                                        word_feature_table.shape[1],
                                        page_features.shape[0]),
                                 dtype=np.float)

    for i in range(word_feature_table.shape[0]): 
        for j in range(word_feature_table.shape[1]): 
            for k in range(page_features.shape[0]):
                calculation_table[i, j, k] = nl.norm(word_feature_table[i, j] - 
                                                     page_features[k])
                
    # TODO: May need to update this score by considering the number of matching components
    return np.min(np.nansum(calculation_table, 2))

    
def find_largest_component_indices(size_table): 
        # size_table = np.empty_like(shape=(len(word_components),
        #                               max_components_per_word),
        #                        dtype=np.float)

    return np.apply_along_axis(np.argmax, 0, size_table)

def calculate_cross_sizes(stats): 
    
    the_table = np.empty(shape=(stats.shape[0],
                                stats.shape[0]),
                         dtype=np.float)
    for i in range(stats.shape[0]):
        the_table[i, :] = stats['area'] / stats['area'][i]
    return the_table

def calculate_cross_distances(stats): 
    the_table = np.empty(shape=(stats.shape[0],
                                stats.shape[0]),
                         dtype=np.float)
    max_area = stats['area'].max()

    for i in range(stats.shape[0]):
        x = stats['cx'][i]
        y = stats['cy'][i]
        the_table[i, :] = (np.abs(stats['cx'] - x) + np.abs(stats['cy'] - y))

    # we divide with the largest component's size, to get more or less an invariant measure. 
    the_table = the_table / max_area
    return the_table

def find_seed_indices(component_features,
                      page_features,
                      n_seeds=100): 

    norm_table = np.empty(shape=(component_features.shape[0],
                                 page_features.shape[0]))
    for i in range(component_features.shape[0]): 
        for j in range(page_features.shape[0]): 
            norm_table[i, j] = nl.norm(component_features[i] -
                                       page_features[j])
    flat_norms = norm_table.flatten()
    flat_norms.sort()
    max_norm_to_consider = flat_norms[n_seeds]
    return min_norms

def find_closest_smaller_elements(cross_sizes,
                                  cross_distances,
                                  candidate_indices,
                                  min_relative_size,
                                  max_relative_distance):
    suitable = np.empty(shape=(candidate_indices.shape[0],
                               cross_sizes.shape[0]),
                        dtype=np.bool)

    for i, ci in enumerate(candidate_indices):
        suitable_by_size = cross_sizes[ci] > min_relative_size
        suitable_by_distance = cross_distances[ci] < max_relative_distance
        suitable[i] = np.bitwise_and(suitable_by_size, suitable_by_distance)
    return suitable

def calculate_word_features(word,
                            feature_extractor,
                            feature_size,
                            nb_epoch):
    word_images = generate_word_images(word)
    word_images = [np.uint8(wi * (255 / wi.max())) 
                   for wi in word_images]

    write_image_list(word_images, "word-images")
    print("Generated {} word images for {}".format(
        len(word_images), word))
    word_components = [aui.text_image_segmentation(wi) 
                       for wi in word_images]
    components_per_word = [len(c) for c in word_components]
    max_components_per_word = max(components_per_word)
    word_feature_table = np.empty(shape=(len(word_components),
                                         max_components_per_word,
                                         feature_size),
                                  dtype=np.float)
    word_feature_table.fill(np.nan)
    size_table = np.empty_like(shape=(len(word_components),
                                      max_components_per_word),
                               dtype=np.float)
    size_table.fill(np.nan)
    distance_table = np.empty_like(shape=(len(word_components),
                                      max_components_per_word),
                                   dtype=np.float)
    distance_table.fill(np.nan)

    for i, wc in enumerate(word_components): 
        wf, bb = wc
        feature_vals = feature_extractor.predict(wf)
        size_vals = bb['area'] / bb['area'].sum()
        distance_vals = (bb['cx'] + bb['cy']) / bb['area'].max()
        word_feature_table[i, :wf.shape[0], :] = feature_vals
        size_table[i, :wf.shape[0]] = size_vals
        distance_table[i, :wf.shape[0]] = distance_vals
    return word_feature_table, size_table, distance_table

def write_image_array(arr, key):
    n_img = arr.shape[0]
    for i in range(n_img):
        cv2.imwrite("/tmp/arr-{}-{}.png".format(key, i), arr[i])

def write_image_list(img_list, key, increase_contrast=False):
    for i, img in enumerate(img_list):
        if increase_contrast:
            img_ = img * (255 / img.max())
        else:
            img_ = img
        cv2.imwrite("/tmp/arr-{}-{}.png".format(key, i), img_)

def merge_page_results(page_results):
    LOG.debug("=== page_results ===")
    LOG.debug(page_results)
    for pr in page_results:
        LOG.debug("=== pr ===")
        LOG.debug(pr)
        LOG.debug("=== pr[0] ===")
        LOG.debug(pr[0])

    total_size = sum([res[0].shape[1] for res in page_results])
    LOG.debug("=== total_size ===")
    LOG.debug(total_size)
    merged_results = np.empty(shape=(total_size,),
                              dtype=[("file", np.int32),
                                     ("x", np.int32),
                                     ("y", np.int32),
                                     ("w", np.int32),
                                     ("h", np.int32),
                                     ("x2", np.int32),
                                     ("y2", np.int32),
                                     ("score", np.float)])
    begin = 0
    for fi, pr in enumerate(page_results):
        res, stats = pr
        LOG.debug("=== stats.shape ===")
        LOG.debug(stats.shape)
        LOG.debug("=== res.shape ===")
        LOG.debug(res.shape)
        end = begin + stats.shape[0]
        LOG.debug("=== (begin, end) ===")
        LOG.debug((begin, end))
        LOG.debug("=== merged_results.shape ===")
        LOG.debug(merged_results.shape)
        merged_results["file"][begin:end] = fi
        LOG.debug("=== merged_results['x'][begin:end].shape ===")
        LOG.debug(merged_results['x'][begin:end].shape)
        merged_results["x"][begin:end] = stats["x"]
        merged_results["y"][begin:end] = stats["y"]
        merged_results["w"][begin:end] = stats["w"]
        merged_results["h"][begin:end] = stats["h"]
        min_results = np.apply_along_axis(np.min, 0, res)
        LOG.debug("=== min_results.shape ===")
        LOG.debug(min_results.shape)
        merged_results["score"][begin:end] = min_results
        begin = end

    merged_results["x2"] = merged_results["x"] + merged_results["w"]
    merged_results["y2"] = merged_results["y"] + merged_results["h"]
    return merged_results


def generate_word_images(word, fonts=None):
    if not fonts:
        fonts = aua.system_arabic_fonts()
    return aui.get_white_on_black_word_images(word,
                                              fonts=fonts,
                                              dpis=list(range(50, 600, 50)))
def load_text_components(imgf):
    artifact_dir = aui.get_artifact_dir(imgf)
    text_components_path = os.path.join(artifact_dir,
                                        "text-components.npz")
    if os.path.exists(text_components_path):
        tcds = np.load(text_components_path)
        return tcds['components'], tcds['stats']
    else:
        adtdv3.main_caller(DEFAULT_TEXT_DETECTION_MODEL,
                           imgf,
                           os.path.join(artifact_dir,
                                        "text-detection-result.jpg"))

def calculate_component_features(components, encoder, feature_size=64):
    LOG.debug("=== components.shape ===")
    LOG.debug(components.shape)
    normalized = components.astype('float32') / 255
    y = encoder.predict(normalized)
    return y


def search_in_image_via_patches(word_patches,
                                img,
                                encoder,
                                patch_size,
                                strides):

    image_layers = aui.color_quantization(img)
    image_layers[image_layers.nonzero()] = 255
    image_layers = image_layers[:, :, :, 0]
    search_tables = []

    for i in range(image_layers.shape[0]):
        LOG.debug("=== image_layers[i].shape ===")
        LOG.debug(image_layers[i].shape)
        # cv2.imwrite("/tmp/image_layers_{}.png".format(i),
        #             image_layers[i])
        dist_img = aui.distance_transform(image_layers[i])
        patches = aui.split_image(dist_img,
                                  patch_size,
                                  patch_size,
                                  strides,
                                  strides)
        LOG.debug("=== patches.shape ===")
        LOG.debug(patches.shape)

        # patches = patches[np.nonzero(
        #     np.apply_over_axes(
        #         np.var, patches, [1, 2]))[0]]
        patches = patches.reshape((patches.shape[0],
                                   1,
                                   patches.shape[1],
                                   patches.shape[2]))
        LOG.debug("=== patches.shape ===")
        LOG.debug(patches.shape)
        patch_features = calculate_component_features(patches,
                                                      encoder,
                                                      patch_size)
        # patch_features = patches
        n_wc = word_patches.shape[0]
        n_ic = patch_features.shape[0]
        search_table = np.empty(shape=(n_wc, n_ic),
                                dtype=np.float)
        for i_w in range(n_wc):
            for i_c in range(n_ic):
                search_table[i_w, i_c] = nl.norm(word_patches[i_w] -
                                                 patch_features[i_c])
        search_tables.append(search_table)

    all_search_tables = np.vstack(search_tables)
    LOG.debug("=== all_search_tables ===")
    LOG.debug(all_search_tables)

    merged_table = np.apply_along_axis(np.min, 0, all_search_tables)
    LOG.debug("=== merged_table ===")
    LOG.debug(merged_table)

    result_img = np.ones(shape=(img.shape[0], img.shape[1]),
                         dtype=np.float) * merged_table.max()

    rs = list(range(0, img.shape[0] // strides))
    cs = list(range(0, img.shape[1] // strides))

    for i, coords in enumerate(itertools.product(rs, cs)):
        r = coords[0]
        c = coords[1]
        x1 = r * strides
        x2 = min(r * strides + patch_size,
                 result_img.shape[0])
        y1 = c * strides
        y2 = min(c * strides + patch_size,
                 result_img.shape[1])
        current_vals = result_img[x1:x2, y1:y2]
        LOG.debug("=== current_vals ===")
        LOG.debug(current_vals)
        new_values = np.ones(shape=(x2-x1, y2-y1),
                             dtype=np.float) * merged_table[i]
        LOG.debug("=== new_values ===")
        LOG.debug(new_values)

        values = np.apply_along_axis(np.min, 2,
                                     np.dstack((current_vals, new_values)))
        LOG.debug("=== values ===")
        LOG.debug(values)

        result_img[x1:x2, y1:y2] = values

    np.save("/tmp/result_img-{}".format(np.random.randint(1000)), result_img)

    return result_img

def search_in_image_via_components(word_features,
                                   img,
                                   feature_extractor,
                                   text_detection_model,
                                   feature_size=64):

    LOG.debug("=== word_features.shape ===")
    LOG.debug(word_features.shape)

    image_layers = aui.color_quantization(img)
    aui.write_image_array(image_layers, "layers")
    image_layers[image_layers.nonzero()] = 255
    image_layers = image_layers[:, :, :, 0]
    dist_tr_layers = [aui.distance_transform(image_layers[i])
                      for i in range(image_layers.shape[0])]
    dist_tr_layers = [np.uint8(wi * (255 / wi.max())) for wi in dist_tr_layers]
    image_component_layers_list = [aui.text_image_segmentation(img, min_area=5, max_area=0.3)
                                   for img in dist_tr_layers]

    image_components_list = [aui.resize_components(img_comp,
                                                   img_stat,
                                                   feature_size,
                                                   feature_size)
                             for img_comp, img_stat in image_component_layers_list]

    search_tables = []
    search_list = []


    for i, component_array in enumerate(image_components_list):
        aui.write_image_array(component_array, str(i))
        ica = component_array.reshape((component_array.shape[0],
                                       1,
                                       component_array.shape[1],
                                       component_array.shape[2]))

        ica_features = calculate_component_features(ica,
                                                    encoder,
                                                    feature_size)

        ica_stats = image_component_layers_list[i][1]

        search_table = np.empty(shape=(img.shape[0], img.shape[1]),
                                dtype=np.float)
        n_wc = word_features.shape[0]
        n_ic = ica_features.shape[0]
        for i_w in range(n_wc):
            for i_c in range(n_ic):
                loc = ica_stats[i_c]
                dist = nl.norm(word_features[i_w] - ica_features[i_c])
                x1, x2 = (loc['x'], loc['x'] + loc['w'])
                y1, y2 = (loc['y'], loc['y'] + loc['h'])
                search_table[x1:x2, y1:y2] = dist
                search_list.append((x1,
                                    x2,
                                    y1,
                                    y2,
                                    loc['area'],
                                    dist))

        search_tables.append(search_table)

    res_img = img.copy()
    LOG.debug("=== search_list ===")
    LOG.debug(search_list)

    sorted_locs = sorted(search_list, key=lambda s: s[5])
    for s in sorted_locs[:10]:
        cv2.rectangle(res_img, (s[2], s[3]), (s[0], s[1]), (255, 0, 0))

    return res_img

def rectangle_coords(stats, width_coeff=1.0, height_coeff=1.0):
    x1 = stats['x']
    x2 = stats['x'] + np.int(stats['w'] * width_coeff)
    y1 = stats['y']
    y2 = stats['y'] + np.int(stats['h'] * height_coeff)
    return (x1, x2, y1, y2)

def main_dispatch(word,
                  images,
                  feature_size,
                  max_results,
                  nb_epoch,
                  fonts,
                  output_dir):
    font_list = fonts.split(',')
    print("Fonts: ", font_list)
    nb_epoch=int(nb_epoch)
    feature_size=int(feature_size)
    max_results=int(max_results)

    search_word_in_library(word,
                           images,
                           fonts=fonts,
                           feature_size=feature_size,
                           max_results=max_results,
                           nb_epoch=nb_epoch,
                           output_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(description="""
    Searches a word in a set of image files
    """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('word',
                        help="The word to be sought")
    parser.add_argument('images',
                        help="Directory that contains the image files")
    parser.add_argument('--output-dir',
                        help="Directory to put the output images",
                        default='/tmp')
    parser.add_argument('--result-size',
                        help="Number of returned results",
                        default=10)
    parser.add_argument('--feature-size',
                        help="Search window size. Smaller means more detailed searches but takes longer. Should be > 32",
                        default=64)
    parser.add_argument('--nb-epoch',
                        help="Number of epochs to train the convolutional autoencoder. Higher means more accurate results.",
                        default=10)
    parser.add_argument('--fonts',
                        help="Comma separated list of font names that will be used to create the search images.",
                        default=','.join(aui.system_arabic_fonts()))
    args = vars(parser.parse_args())
    main_dispatch(**args)

if __name__ == '__main__':
    main()
