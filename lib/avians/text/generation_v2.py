
import scipy
import scipy.spatial as spat
import os.path
import numpy as np
import pandas as pd
import re
import itertools
import heapq
import pickle
import logging
LOG = logging.getLogger(__name__)

import harfbin.util.arabic_defs as huod
import harfbin.util.converter as huc
import harfbin.similarity.comparator as hsc

MAX_COMPONENTS_PER_LINE = 500
MAX_PARSE               = 1000
CWE                     = 100
PWE                     = 200

KAPPA_DICT              = 10
KAPPA_DIST              = 5
KAPPA_BIGRAM            = 4

ARABIC_CORPUS_FILE = os.path.expandvars("$HOME/Repository/dervaze/doc/leylamecnun-corpus.txt")
VISENC_ADDITIONS = os.path.expandvars("$HOME/Repository/harfbin/visenc-supplied.txt")
ARABIC_DICT_SET_FILE = os.path.expandvars("$HOME/Repository/dervaze/doc/leylamecnun-words.pickle")
ARABIC_BIGRAM_FILE = os.path.expandvars("$HOME/Repository/dervaze/doc/leylamecnun-bigrams.pickle")

SMOOTHING_LENGTH_FACTOR = 20
SMOOTHING_VARIANCE_FACTOR = 200

PROB_SCALE=1000.0

def generate_word_dict_set(arabic_text): 
    """Generates a word list from a piece of text by delimiting from the whitespace. 

    It returns a dictionary. The keys of the dictionary are visenc without
    diacritics and values of the dictionary are sets of words. So we can check
    them quickly.
    """
    
    visenc_words = [huc.arabic_to_visenc(w.strip()) for w in arabic_text.split()]

    words = {}
    for w in visenc_words:
        canonical = huc.canonical_visenc(w)
        if canonical not in words: 
            words[canonical] = set()
        words[canonical].add(w)
    LOG.debug("=== words[:10] ===")
    LOG.debug(list(words.keys())[:10])
    return words

def generate_bigram_set(arabic_text):
    visenc_words = [huc.arabic_to_visenc(w.strip()) for w in arabic_text.split()]
    bigrams = {}
    for i, w in enumerate(visenc_words[:-1]):
        n = visenc_words[i+1]
        k = (w, n)
        if k not in bigrams: 
            bigrams[k] = 1
        else: 
            bigrams[k] += 1

    return bigrams

def divide_into_letters(label):
    """Shards a label string into individual visual encoding letters.

    :param label: String to be decomposed
    :rtype: [string]
    """

    begin_index = 0
    result_list = []

    while begin_index < len(label):
        match = re.search(huod.VISENC_REGEX, label[begin_index:])
        if match is None:
            break
        else:
            result_list.append(match.group(0))  # (location[0]+location[2]/2)

            begin_index += len(match.group(0))

    return result_list

def generate_location_kdtree(component_list): 
    "Returns a midpoint kdtree and midpoint to component dictionary"
    midpoints = {c.location[0:2]: c.key() for c in component_list}
    kdtree = spat.KDTree(data=midpoints.keys())
    return kdtree, midpoints

def find_closest_letters(component_table, stem_i, diacritic_i): 
    "Sort the letters in the stem by their proximity to the diacritic"

    stem = component_table[stem_i]
    diacritic = component_table[diacritic_i]

    if component_table[stem_i]["label_length"] == 1: 
        return str(stem["label"])

    # We divide the whole rectangle into equal regions 
    # This is simplistic, but deciding per letter is too complex to benefit
    cy = stem["cy"]
    ll = stem["label_length"]
    incr_x = float(stem["w"]) / ll
    letter_mid = incr_x / 2
    cxs = [i * incr_x for i in range(0, ll)] + stem["x"] + letter_mid
    letters = stem["letters"]
    assert len(cxs) == len(letters)
    loc_letter = [(cx, cy, l) for cx, l in zip(cxs, letters)]
    dx = diacritic["cx"]
    dy = diacritic["cy"]
    # we use manhattan distance
    sortf = lambda s, d: abs(s[0] - dx + s[1] - dy)
    loc_letter = sorted(loc_letter, key=sortf)
    return [ll[2] for ll in loc_letter]


def calculate_minimum_size_to_consider(bounding_box_arr):
    """
    :param bounding_box_arr: An array with dtype
    [('x', np.int), ('y', np.int), ('w', np.int), ('h', np.int)]

    """
    size_arr = bounding_box_arr['w'] * bounding_box_arr['h']
    mean_size = np.mean(size_arr)
    stdev_size = np.std(size_arr)
    min_size_to_consider = mean_size - 2 * stdev_size
    return min_size_to_consider

def vertical_stem_histogram(bounding_box_arr):
    """
    :param bounding_box_arr: An array with dtype
    [('x', np.int), ('y', np.int), ('w', np.int), ('h', np.int)]


    """
    min_size_to_consider = calculate_minimum_size_to_consider(bounding_box_arr)
    max_y = 
    histogram = np.array([0] * max_y)
    for c in component_list:
        c_x, c_y, c_w, c_h = c.location
        if c_w * c_h >= min_size_to_consider: 
            histogram[(c_y):(c_y + c_h)] += c_w
    return histogram

def vertical_component_histogram(component_list):
    max_y = max([c.y + c.height for c in component_list])
    LOG.debug("=== max_y ===")
    LOG.debug(max_y)
    histogram = np.array([0] * max_y)
    for c in component_list:
        c_x, c_y, c_w, c_h = c.location
        LOG.debug("=== c.location ===")
        LOG.debug(c.location)
        histogram[(c_y):(c_y + c_h)] += c_w
    LOG.debug("=== histogram ===")
    LOG.debug(histogram)
    return histogram

def find_line_peaks(component_list): 
    """Find line locations by peaks in vertical histogram"""
    component_hist = vertical_stem_histogram(component_list)
    max_y = component_hist.shape[0]
    # LOG.debug("=== max_y ===")
    # LOG.debug(max_y)
    
    # LOG.debug("=== component_hist ===")
    # LOG.debug(component_hist)
    
    smoothing_filter = scipy.signal.gaussian(len(component_hist)//SMOOTHING_LENGTH_FACTOR,
                                             len(component_hist)//SMOOTHING_VARIANCE_FACTOR)
    comp_hist_convolved = np.convolve(component_hist, smoothing_filter)
    peaks = (np.diff(np.sign(np.diff(comp_hist_convolved))) < 0).nonzero()[0] + 1  # local max
    # for p in list(reversed(peaks)):
    #     plt.plot((0,self.max_x), (p,p), "k",  lw=1)
    # plt.show()
    logging.debug("Peaks: {}".format(peaks))
    if len(peaks) > 0:
        return list(reversed(peaks))
    else:
        # if no lines found, return a line in the middle
        return [len(component_hist) / 2]


def __find_max(component_list):
    x_list = []; y_list = []
    for c in component_list:
        c_x, c_y, c_w, c_h = c.location
        x_list = np.append(x_list, c_x+c_w)
        y_list = np.append(y_list, c_y+c_h)
    return max(x_list), max(y_list)

def find_lines(component_list): 
    """Finds the peaks and associates each component with a line by comparing y
    coordinates of each components and lines. Returns a dict
            {line_id:[component1,component2...],...}
    """

    peaks = find_line_peaks(component_list)
    # min_size_to_consider = calculate_minimum_size_to_consider(component_list)
    line_dict = {}
    # rect_dict = {}
    # put large elements first
    for c in component_list:
        nearest = (sorted(peaks, key=lambda p: abs(p - (c.y + c.h / 2))))[0]
        if nearest not in line_dict: 
            line_dict[nearest] = []
        line_dict[nearest].append(c)
    #     rect_dict[c.location] = nearest
    # # attach smaller elements by closest large element 
    # rects = list(rect_dict.keys())
    # for c in component_list: 
    #     if c.width * c.height <= min_size_to_consider: 
    #         nearest = (sorted(rects, key=lambda r: abs(r[2] - (c.y + c.h / 2))))[0]
    #     # LOG.debug("=== rect_dict ===")
    #     # LOG.debug(rect_dict)
    #     p = rect_dict[nearest]
    #     line_dict[p].append(c)

    return line_dict

def sort_by_mid_x(component_list, reverse=False): 
    return sorted(component_list, key=lambda c: c.x + c.width / 2, reverse=reverse)

def sort_by_mid_y(component_list, reverse=False): 
    return sorted(component_list, key=lambda c: c.y + c.height / 2, reverse=reverse)

def sort_by_x(component_list, reverse=False): 
    if reverse: 
        return sorted(component_list, key=lambda c: c.x + c.width, reverse=True)
    else: 
        return sorted(component_list, key=lambda c: c.x, reverse=False)

def sort_from_right_to_left(component_list):
    return sort_by_x(component_list, True)

def find_baseline(component_list): 
    y_list = [c.y for c in component_list]
    max_y = max(y_list)
    min_y = min(y_list)
    baseline = min_y + (max_y - min_y) / 2
    return baseline

def all_variations(component_list, up_to=1):
    "Produces a cartesian product of all line reading variations."
    # Each element in the component_list
    line_labels = []
    prev_rect = (float("inf"), float("inf"), 
                 float("inf"), float("inf"))
    # [[(label, dist), (label|, dist)]]
    # LOG.debug("=== len(component_list) ===")
    # LOG.debug(len(component_list))

    # array_w = len(component_list)
    # array_h = 2 * np.product([min(len(c.label), up_to) for c in component_list])

    # variations_array = np.zeros(shape=(array_w, array_h), dtype=[("label", "S20"),
    #                                                              ("x", np.int),
    #                                                              ("y", np.int),
    #                                                              ("w", np.int),
    #                                                              ("h", np.int),
    #                                                              ("label_dissimilarity", np.double)])

    # LOG.debug("=== variations_array.shape ===")
    # LOG.debug(variations_array.shape)
    
    # va_i = 0


    baseline = find_baseline(component_list)
    LOG.debug("=== baseline ===")
    LOG.debug(baseline)
    
    for c in component_list:
        label_dists = [(l, min(dist_list)) for l, dist_list in c.label.items()]
        sorted_labels = sorted(label_dists, key=lambda x: x[1])
        considered_labels = sorted_labels[:up_to]
        label_variations = []
        for l, d in considered_labels: 
            # if it's a diacritic: 
            if is_diacritic(l): 
                if c.y > baseline: 
                    l = "u" + l
                else: 
                    l = "o" + l
            # If it can end a word or not, we consider two different versions
            elif re.match(huod.VISENC_POSSIBLE_WORD_END, l):
                label_variations.append((l + "+", d))
            label_variations.append((l, d))
        line_labels.append(label_variations)
        # LOG.debug("=== label_variations ===")
        # LOG.debug(label_variations)

    return itertools.product(*line_labels)

def calculate_word_score(word_dict_set, line_variation):
    """
    The input is single output of variations, namely,

    [(label1, dist), (label2, dist2), ...]

    If label ends with +, it means it should be merged with the next label for word. 
    """
    LOG.debug("=== line_variation ===")
    LOG.debug(line_variation)
    LOG.debug("=== line_variation[0] ===")
    LOG.debug(line_variation[0])
    
    
    whole_line = merge_stems_diacritics(line_variation)
    words = whole_line.split()
    score = 0

    for w in words:
        cw = huc.canonical_visenc(w)
        
        if cw in word_dict_set: 
            LOG.debug("=== Found: (w, cw) ===")
            LOG.debug((w, cw))
            score += len(w)
            
    return score

def calculate_component_score(line_variation): 
    "Calculates the total object similarity score"

    MAX_DISSIM = 100.0
    total_dissim = np.sum([(MAX_DISSIM - d) for l, d in line_variation])
    return total_dissim / (len(line_variation) * MAX_DISSIM) 

def select_best_readings(word_dict_set, component_list, up_to=1, at_most=10): 
    """
    Creates a list of candidates, sorts them with the word score and returns the best readings with score
"""
    readings = []
    variations = all_variations(component_list, up_to)
    LOG.debug("=== variations ===")
    LOG.debug(variations)
    
    for v in variations:
        # LOG.debug("=== v ===")
        # LOG.debug(v)
        
        word_score = calculate_word_score(word_dict_set, v)
        LOG.debug("=== word_score ===")
        LOG.debug(word_score)
        # component_score = calculate_component_score(v)
        # LOG.debug("=== component_score ===")
        # LOG.debug(component_score)
        
        # reading_score = np.sqrt(word_score * word_score + 
        #                         component_score * component_score)
        reading_score = word_score
        # LOG.debug("=== reading_score ===")
        # LOG.debug(reading_score)
        heapq.heappush(readings, (-1 * reading_score, v))
        # LOG.debug("=== readings ===")
        # LOG.debug(readings)

        # Heap Cleanup 
        if len(readings) > 10 * at_most: 
            new_readings = []
            for i in range(at_most): 
                heapq.heappush(new_readings,
                               heapq.heappop(readings))
            readings = new_readings

    results = []
    for i in range(min(at_most, len(readings))): 
        s, r = heapq.heappop(readings)
        results.append((-1 * s, r))
    return results

def visenc_additions(word_dict_set):
    f = open(VISENC_ADDITIONS)
    visenc_lines = f.readlines()
    for l in visenc_lines:
        if len(l.strip()) > 0: 
            items = l.split(":")
            k = huc.canonical_visenc(items[0].strip())
            if k not in word_dict_set: 
                word_dict_set[k] = set()
            word_dict_set[k].add(items[1].strip())

    return word_dict_set

def load_or_generate_word_dict_set(): 
    if os.path.exists(ARABIC_DICT_SET_FILE): 
        LOG.debug("=== ARABIC_DICT_SET_FILE found===")
        LOG.debug(ARABIC_DICT_SET_FILE)
        f = open(ARABIC_DICT_SET_FILE, "rb")
        word_dict_set = pickle.load(f)
    else: 
        LOG.debug("=== ARABIC_DICT_SET_FILE not found===")
        LOG.debug(ARABIC_DICT_SET_FILE)
        
        text_f = open(ARABIC_CORPUS_FILE)
        arabic_text = text_f.read()
        word_dict_set = generate_word_dict_set(arabic_text)
        text_f.close()

        pf = open(ARABIC_DICT_SET_FILE, mode="w+b")
        pickle.dump(word_dict_set, pf)
        pf.close()

    word_dict_set = visenc_additions(word_dict_set)
    
    return word_dict_set

def merge_stems_diacritics(label_dist_list): 
    LOG.debug("=== label_dist_list ===")
    LOG.debug(label_dist_list)
    whole_line = " ".join([lab for lab, d in label_dist_list])
    # delete space before u, i and o
    whole_line = re.sub(r' ([uoi])', '\g<1>', whole_line)
    # delete space after the words containing + sign
    whole_line = re.sub(r'([^ ]+)\+([^ ]*) ', '\g<1>\g<2>', whole_line)
    return whole_line

def build_line(line, word_dict): 
    items = line[1]
    merged = merge_stems_diacritics(line)
    split_words = merged.split()
    collect_words = []
    for sw in split_words: 
        cw = huc.canonical_visenc(sw)
        if cw in word_dict: 
            ss = word_dict[cw]
            # select the most similar to sw
            min_score = float("inf")
            min_el = ""
            for s in ss: 
                word_sim = hsc.levenshtein_distance(s, sw)
                if word_sim < min_score: 
                    min_score = word_sim
                    min_el = s
            collect_words.append(min_el)
        else: 
            collect_words.append(sw)

    LOG.debug("=== collect_words ===")
    LOG.debug(collect_words)
    
    return " ".join(collect_words)

def component_line_to_text(word_dict_set, components): 
    sorted_components = sort_from_right_to_left(components)
    LOG.debug("=== sorted_components ===")
    LOG.debug(sorted_components)
    
    best_readings = select_best_readings(word_dict_set, 
                                         sorted_components, 
                                         up_to=1)
    LOG.debug("=== best_readings ===")
    LOG.debug(best_readings)
    
    # currently we return the top element simply. 
    LOG.debug("=== best_readings[0] ===")
    LOG.debug(best_readings[0])
    
    return build_line(best_readings[0][1], word_dict_set)

def greedy_label_list(component_list): 
    ll = []
    baseline = find_baseline(component_list)
    for c in component_list: 
        l = c.min_label()
        if c.is_diacritic(): 
            if c.y > baseline: 
                l = "u" + l
            else: 
                l = "o" + l
        ll.append(l)
    return ll
            
def greedy_component_line_to_text(word_dict_set, component_list): 
    sorted_components = sort_from_right_to_left(component_list)
    label_list = greedy_label_list(sorted_components)
    canon_elements = []

    begin = 0
    while begin < len(label_list): 
        found = False
        for end in range(min(begin + 10, len(label_list)), begin, -1):
            sublist = label_list[begin:end]
            merged = "".join(sublist)
            LOG.debug("=== merged ===")
            LOG.debug(merged)
            canon_vis = huc.canonical_visenc(merged)
#            print(canon_vis)
            LOG.debug("=== canon_vis ===")
            LOG.debug(canon_vis)
            if canon_vis in word_dict_set:
                LOG.debug("=== Found ===")
                canon_elements.append(canon_vis)
                # LOG.debug("=== label_list[end-1] ===")
                # LOG.debug(label_list[end-1])
                # LOG.debug("=== label_list[end] ===")
                # LOG.debug(label_list[end])
                # LOG.debug("=== label_list[end+1] ===")
                # LOG.debug(label_list[end+1])
                begin = end
                if begin < len(label_list): 
                    LOG.debug("=== new label_list[begin] ===")
                    LOG.debug(label_list[begin])
                found = True
                break
            else:
                LOG.debug("=== Not Found ===")

        if not found:
            # skip until the next stem
            current_element = label_list[begin] + '!'
            begin += 1
            while begin < len(label_list) and not is_stem(label_list[begin]): 
            # LOG.debug("=== label_list[begin] ===")
            # LOG.debug(label_list[begin])
                current_element += "-" + label_list[begin]
                begin += 1
            # LOG.debug("=== current_element ===")
            # LOG.debug(current_element)
            canon_elements.append(current_element)

    # print(canon_elements)
    return canon_elements

def canon_to_text(canon_elements, word_dict_set): 

    converted = []

    for c in canon_elements: 
        if c in word_dict_set: 
            s = word_dict_set[c]
            converted.append(list(s)[0])
        else: 
            stem = c.split('!')[0]
            # stem = c
            converted.append(stem)
    return " ".join(converted)

def component_list_to_text(component_list): 
    lines = find_lines(component_list)
    line_labels = {k: [c.min_label() for c in comp_list] for k, comp_list in lines.items()}
    LOG.debug("=== line_labels ===")
    LOG.debug(line_labels)
    
    generated_dict = {}
    word_dict_set = load_or_generate_word_dict_set()
    bigram_dict = generate_bigram_set(ARABIC_CORPUS_FILE)
    
    for line, components in lines.items(): 
        generated_dict[line] = greedy_component_line_to_text(word_dict_set, components)

    generated_lines = [canon_to_text(generated_dict[k],
                                     word_dict_set)
                       for k in sorted(list(generated_dict))]

    generated_text = '\n'.join(generated_lines)
    LOG.debug("=== generated_text ===")
    LOG.debug(generated_text)
    return generated_text

def component_lib_to_text(component_lib):
    cl = [c for k, c in component_lib.items()]
    # LOG.debug("=== cl ===")
    # LOG.debug(cl)
    return component_list_to_text(cl)

##### DISCARDED CODE #########
# The following code is used before but not used anymore
# We may find value in the coming times, but as of now, no need for that. 


def build_label_table(component_list): 
    """This builds the whole data structure that line detection, assoaciation and text generation runs.

    Label table is a pandas dataframe. It has a set of columns to keep all
    attributes of a component and also some more to aid in running the
    algorithms. """
    data_dict = {}

    c_id = 0
    for k, c in self.items():
        c_id += 1
        for lab, dist_list in c.label: 
            data_dict[k] = {}
            data_dict[k]["component_id"] = c_id
            data_dict[k]["key"] = k
            data_dict[k]["docname"] = c.image_filename
            data_dict[k]["label"] = lab
            data_dict[k]["label_length"] = len(lab)
            data_dict[k]["dist"] = min(dist_list)
            data_dict[k]["prob"] = -1 # Sometimes we have prob, sometimes dist
            data_dict[k]["x"] = c.x
            data_dict[k]["y"] = c.y
            data_dict[k]["w"] = c.width
            data_dict[k]["h"] = c.height
            data_dict[k]["is_stem"] = c.is_stem()
            data_dict[k]["cx"] = c.x + c.w // 2
            data_dict[k]["cy"] = c.y + c.h // 2
            data_dict[k]["letter_list"] = divide_into_letters(lab)
            # NOT YET FILLED 
            data_dict[k]["line_id"] = -1  # The line that this resides
            data_dict[k]["linked_stem"] = -1  # If this is a diacritic, the stem that's associated with
            data_dict[k]["modified_label"] = ""  # e.g. "u1" for diacritic 1, or no diac label for stems. (sometimes that have stems.)

    dataframe = p.DataFrame.from_dict(data_dict,
                                      orient='index')
    return dataframe

def dist_to_prob(dist):
    "Converts a dist value to a prob by dividing it with PROB_SCALE"
    m = np.min(np.abs(dist_list))
    return 1 - m if m < 1 else 0

def prob_to_dist(prob):
    "Converts a probability value to dist by multiplying it with PROB_SCALE"
    return (1 - prob) * PROB_SCALE

def init_tables(): 
    the_table = np.zeros(shape=(MAX_COMPONENTS_PER_LINE, MAX_PARSE), 
                         dtype=[('label', 'U20'),  # Label of the component
                                # One of the following values are used, up to classification method
                                ('dist', float), # Distance value, from classification
                                ('prob', float), # Probability value, from classification
                                # Location on the page
                                ('x', int), 
                                ('y', int), 
                                ('w', int),
                                ('h', int),
                                ('cx', int), # x of center point
                                ('cy', int), # y of center point
                                # The marker that either represents PWE or CWE
                                ('marker', int),
                                # The scores for this element. 
                                ('dict_score', float),
                                ('ngram_score', float)])

    # Each column 
    column_indexes = np.zeros(shape=(MAX_PARSE, ), dtype=np.int)
    return the_table, column_indexes
    
def stem_diac(stem, stem_loc, diacritic, diacritic_loc): 
    "Returns a list of labels that is possible combining stem and diacritic"
    letter_list = divide_into_letters(stem)

    def location_above(loc_a, loc_b):
        LOG.debug("=== loc_a ===")
        LOG.debug(loc_a)
        LOG.debug("=== loc_b ===")
        LOG.debug(loc_b)
        
        return (loc_a[1] - loc_b[1] <= 0)  # we used <= because we reversed lines and diacritics&stems locations reversed also

    if location_above(diacritic_loc, stem_loc):
        pseudo_diacritic = "o" + diacritic
    else:
        pseudo_diacritic = "u" + diacritic

    possible_combinations = []
    for current_letter_index, current_letter in enumerate(letter_list):
        if (current_letter + pseudo_diacritic) in huod.VISENC_POSSIBLE_MERGES: 
            prefix = "".join(letter_list[:current_letter_index])
            current_letter_combined = current_letter + pseudo_diacritic 
            postfix = "".join(letter_list[current_letter_index + 1:])
            possible_combinations.append(prefix + current_letter_combined + postfix)

    return list(set(possible_combinations))

def step_merge_diacritic(p_table, p_ci_t, p_row_i,
                         c_table, c_ci_t, c_row_i, 
                         diacritic, diacritic_loc, distance, prob):
    label_index = p_ci_t[p_row_i]
    # The possible compositions of this diacritic and stem
    stem_diac_productions = stem_diac(p_table["label"][p_row_i, label_index], 
                                      p_table[["x", "y", "w", "h", "cx", "cy"]][p_row_i, label_index],
                                      diacritic, diacritic_loc)

    for new_label in stem_diac_productions: 
        # Copy all labels and other info
        c_table[c_row_i, :] = p_table[p_row_i, :]
        # The new label is the composite version of label and diacritic
        c_table[c_row_i, label_index]["label"] = new_label
        # We don't increase the index, no new additions to the row, just a diacritic merge
        c_ci_t[c_row_i] = p_ci_t[p_row_i]
        # This row is OK
        c_row_i += 1
    return c_table, c_ci_t, c_row_i

def step_initial_stem(p_table, p_ci_t, p_row_i,
                  c_table, c_ci_t, c_row_i, 
                  stem, stem_loc, distance, prob):
    # Add this as an initial element, without copying anything from previous
    new_ci = 0
    c_table[c_row_i, new_ci]["label"] = stem
    c_table[c_row_i, new_ci]["dist"] = distance
    c_table[c_row_i, new_ci]["prob"] = prob
    c_table[c_row_i, new_ci]["x"] = stem_loc[0]
    c_table[c_row_i, new_ci]["y"] = stem_loc[1]
    c_table[c_row_i, new_ci]["w"] = stem_loc[2]
    c_table[c_row_i, new_ci]["h"] = stem_loc[3]
    c_table[c_row_i, new_ci]["cx"] = (stem_loc[0] + stem_loc[2]) // 2
    c_table[c_row_i, new_ci]["cy"] = (stem_loc[1] + stem_loc[3]) // 2
    # The condition to put a CWE marker is the absence of stem breaking letters at the end. 
    if re.match(huod.VISENC_WORD_END, c_table[c_row_i, new_ci]["label"]):
        new_marker = CWE
    else: 
        new_marker = PWE
    c_table[c_row_i, new_ci]["marker"] = new_marker
    return c_table, c_ci_t, c_row_i



def step_add_stem(p_table, p_ci_t, p_row_i,
                  c_table, c_ci_t, c_row_i, 
                  stem, stem_loc, distance, prob):
    # We add this stem as a new element
    # Copy previous line
    c_table[c_row_i, :] = p_table[p_row_i, :]

    # Update the new label
    c_ci_t[c_row_i] = p_ci_t[p_row_i] + 1
    new_ci = c_ci_t[c_row_i]
    c_table[c_row_i, new_ci]["label"] = stem
    c_table[c_row_i, new_ci]["dist"] = distance
    c_table[c_row_i, new_ci]["prob"] = prob
    c_table[c_row_i, new_ci]["x"] = stem_loc[0]
    c_table[c_row_i, new_ci]["y"] = stem_loc[1]
    c_table[c_row_i, new_ci]["w"] = stem_loc[2]
    c_table[c_row_i, new_ci]["h"] = stem_loc[3]
    c_table[c_row_i, new_ci]["cx"] = (stem_loc[0] + stem_loc[2]) // 2
    c_table[c_row_i, new_ci]["cy"] = (stem_loc[1] + stem_loc[3]) // 2
    # The condition to put a CWE marker is the absence of stem breaking letters at the end. 
    if re.match(huod.VISENC_WORD_END, c_table[c_row_i, new_ci]["label"]):
        new_marker = CWE
    else: 
        new_marker = PWE
    c_table[c_row_i, new_ci]["marker"] = new_marker
    return c_table, c_ci_t, c_row_i


def merge_labels(the_row, second_stem): 
    "Merges two stems together to be considered a single word"
    the_row["label"] = the_row["label"] + second_stem

def merge_dists(the_row, dist_prob_new): 
    """Merges two dist/prob values into a single one. Currently it simply takes the
mean of two values.
    """
    the_row["dist"] = (the_row["dist"] + dist_prob_new[0]) / 2
    the_row["prob"] = (the_row["prob"] + dist_prob_new[1]) / 2

def merge_locs(the_row, loc_new): 
    "Merges two location tuple into two. loc_orig is 6-tuple and loc_new is 4-tuple. loc_orig contains center point in addition of x, y, width and height values."
    
    x1 = the_row["x"] 
    y1 = the_row["y"] 
    w1 = the_row["w"]
    h1 = the_row["h"]
    x2, y2, w2, h2 = loc_new
    # New x point is the minimum of x1 and x2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(w1, w2)
    h = max(h1, h2)
    cx = x + w // 2
    cy = y + h // 2
    the_row["x"] = x
    the_row["y"] = y
    the_row["w"] = w
    the_row["h"] = h
    the_row["cx"] = cx
    the_row["cy"] = cy

def step_merge_stem(p_table, p_ci_t, p_row_i,
                    c_table, c_ci_t, c_row_i,
                    stem, stem_loc, dist, prob):
    # Copy the previous
    c_table[c_row_i, :]  = p_table[p_row_i, :]
    label_index = p_ci_t[p_row_i]
    merge_labels(c_table[c_row_i, label_index], stem)
    merge_dists(c_table[c_row_i, label_index], (dist, prob))
    merge_locs(c_table[c_row_i, label_index], stem_loc)
    if re.match(huod.VISENC_WORD_END, c_table[c_row_i, label_index]["label"]): 
        new_marker = CWE
    else: 
        new_marker = PWE
    c_table[c_row_i, label_index]["marker"] = new_marker
    # There is no change in c_ci_t, as we didn't add anything
    c_row_i += 1
    return c_table, c_ci_t, c_row_i

def is_diacritic(lab): 
    return lab.isdigit()

def is_stem(lab): 
    m = re.match(r'^' + huod.VISENC_STEM_BEGIN_REGEX, lab)
    if m == None: 
        return False
    else: 
        return True

def is_number(lab): 
    return (lab[0] == "n") if len(lab) > 1 else False

def fill_tables(component_list): 
    # LOG.debug("*** fill_tables ***")
    # LOG.debug("=== component_list ===")
    # LOG.debug(component_list)
    # we have two sets of tables, one for current iteration, one for previous
    p_table, p_ci_t = init_tables()
    # LOG.debug("=== initial p_* tables ===")
    # LOG.debug(p_table)
    # LOG.debug(p_ci_t)
    c_table, c_ci_t = init_tables()
    # LOG.debug("=== initial c_* tables ===")
    # LOG.debug(c_table)
    # LOG.debug(c_ci_t)
    p_row_i = -1
    c_row_i = 0

    for c in component_list:
        # make current previous and use previous for the current to save space
        s_table, s_ci_t = p_table, p_ci_t
        p_table, p_ci_t = c_table, c_ci_t
        c_table, c_ci_t = s_table, s_ci_t
        # We begin from the first row again for each component
        p_row_i = c_row_i
        c_row_i = 0

        # LOG.debug("=== for component {} ===".format(c))
        # LOG.debug("labels: {}".format(c.label))
        # LOG.debug("p_row_i: {}".format(p_row_i))
        
        for l, dists in c.label.items():
            d = np.min(dists)
            # LOG.debug("Label: {} Distance: {}".format(l, d))
            # LOG.debug("=== BEFORE c_table[c_row_i, :] ===")
            # LOG.debug(c_table[c_row_i, :])
            if is_diacritic(l): 
                # Add diacritic label to all 
                LOG.debug("{} is diacritic".format(l))
                for i in range(0, p_row_i + 1):
                    # We make the call with probability = 0 when we use dist only

                    c_table, c_ci_t, c_row_i = step_merge_diacritic(p_table, p_ci_t, p_row_i,
                                                                    c_table, c_ci_t, c_row_i,
                                                                    l, c.location, d, 0)
            elif is_stem(l): 
                # LOG.debug("{} is stem".format(l))
                for i in range(0, p_row_i + 1): 
                    # We need to check if the last element is CWE or PWE
                    # LOG.debug("=== p_table[p_row_i, p_ci_t[p_row_i]] ===")
                    # LOG.debug(p_table[p_row_i, p_ci_t[p_row_i]])
                    if p_table[p_row_i, p_ci_t[p_row_i]]["marker"] == CWE:
                        # LOG.debug("Last element is CWE")
                        c_table, c_ci_t, c_row_i = step_add_stem(p_table, p_ci_t, p_row_i,
                                                                 c_table, c_ci_t, c_row_i,
                                                                 l, c.location, d, 0)

                    elif p_table[p_row_i, p_ci_t[p_row_i]]["marker"] == PWE: 
                        # We'll add two rows, one containing oldlabel+newlabel as a single word, one as two separate words
                        # LOG.debug("Last element is PWE")
                        # Situation 1: Both stems are members of the same word
                        c_table, c_ci_t, c_row_i = step_merge_stem(p_table, p_ci_t, p_row_i,
                                                                   c_table, c_ci_t, c_row_i, 
                                                                   l, c.location, d, 0)
                        
                        # Situation 2: Stems are members of different words
                        # The case is same of the above, CWE case
                        c_table, c_ci_t, c_row_i = step_add_stem(p_table, p_ci_t, p_row_i,
                                                                 c_table, c_ci_t, c_row_i,
                                                                 l, c.location, d, 0)
                    else: 
                        # If neither, it may be an initial element. 
                        # LOG.debug("Initial element")
                        c_table, c_ci_t, c_row_i = step_initial_stem(p_table, p_ci_t, p_row_i,
                                                                     c_table, c_ci_t, c_row_i,
                                                                     l, c.location, d, 0)
                        
            elif is_number(l):
                # LOG.debug("{} is number".format(l))
                for i in range(0, p_row_i + 1): 
                     # If the last label is also a number, merge, otherwise add. In both cases new marker is a CWE
                    if is_number(p_labels_t[p_row_i, p_index_t[i] - 1]): 
                        LOG.debug("Last element is number")
                        c_table, c_ci_t, c_row_i = step_merge_stem(p_table, p_ci_t, p_row_i,
                                                                   c_table, c_ci_t, c_row_i, 
                                                                   l, c.location, d, 0)
                    else: 
                        LOG.debug("Last element is not a number")
                        c_table, c_ci_t, c_row_i = step_add_stem(p_table, p_ci_t, p_row_i,
                                                                 c_table, c_ci_t, c_row_i,
                                                                 l, c.location, d, 0)
                    
            else: 
                # LOG.debug("{} is neither diacritic, nor stem, nor number")
                # We currently don't do anything
                pass
                # # Add the label as an ornament. This makes the previous marker CWE
                # c_labels_t, c_dist_t, c_index_t, c_row_i = step_add_ornament(p_labels_t, p_dist_t, p_index_t, i,
                #                                                              c_labels_t, c_dist_t, c_index_t, c_row_i,
                #                                                              l, d)

            # LOG.debug("=== AFTER c_table[c_row_i, :] ===")
            # LOG.debug(c_table[c_row_i, :])

    return c_table, c_ci_t


def word_score(prospective_word): 
    p_str = str(prospective_word)
    if (WORD_LIST.visenc == p_str).any():
        return len(p_str) / 2
    else: 
        return 0

def calculate_dict_score(table_row): 
    LOG.debug("=== table_row ===")
    LOG.debug(table_row)
    
    total_score = 0.0
    for word in np.nditer(table_row): 
        LOG.debug("=== word ===")
        LOG.debug(word)
        LOG.debug("=== type(word) ===")
        LOG.debug(type(word))
        
        total_score += word_score(word["label"])
    return total_score

def calculate_dist_score(table_row):
    total_score = 0.0
    for word in np.nditer(table_row): 
        total_score += word["dist"]
    # dist_score is negative, the lower, the better
    return -1 * total_score

def calculate_ngram_score(table_row): 
    "We don't calculate this as of now."
    return 0

def calculate_line_scores(the_table, ci_t):
    
    indices_to_consider = np.nonzero(ci_t > 0)
    scores_t = np.ones(shape=(4, MAX_PARSE), dtype=np.float) * float("-inf")
    LOG.debug("=== indices_to_consider ===")
    LOG.debug(indices_to_consider)
    
    for i in np.nditer(indices_to_consider): 
        row = the_table[i, 0:ci_t[i]]
        LOG.debug("=== row ===")
        LOG.debug(row)
        scores_t[0, i] = calculate_dict_score(row)
        scores_t[1, i] = calculate_dist_score(row)
        scores_t[2, i] = calculate_ngram_score(row)
    
    scores_t[3, :] = scores_t[0, :] + scores_t[1, :] + scores_t[2, :]
    return scores_t[3, :]
        
