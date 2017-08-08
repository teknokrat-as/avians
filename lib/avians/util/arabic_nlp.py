
from avians.util import *
import avians.util.arabic as aua
import avians.util.image as aui

from enum import Enum
import codecs

class Direction(Enum):
    UNDER = 1
    OVER  = 2
    RIGHT = 3
    LEFT  = 4


def _array_x2(a):
    z = np.zeros(shape=a.shape,
                 dtype=a.dtype)
    return np.concatenate((a, z))

def _add_component_table_record(t, c1, c2, direction, diacritic, distance, current_table_index):
    current_rec = np.where(t['c1'] == c1
                           and t['c2'] == c2
                           and t['direction'] == int(direction.value)
                           and t['diacritic'] == diacritic
                           and t['distance'] == distance)
    if t[current_rec]:
        t[current_rec]['frequency'] += 1
    else:
        if current_table_index >= t.shape[0]:
            t = _array_x2(t)
        t[current_table_index] = (c1, c2, int(direction.value), diacritic, distance, 1)
        current_table_index += 1
    return (t, current_table_index)

def _add_word_table_record(t, w1, w2, direction, distance, current_table_index):
    current_rec = np.where(t['w1'] == w1
                           and t['w2'] == w2
                           and t['direction'] == int(direction.value)
                           and t['distance'] == int(distance.value))
    if t[current_rec]:
        t[current_rec]['frequency'] += 1
    else:
        if current_table_index >= t.shape[0]:
            t = _array_x2(t)
        t[current_table_index] = (w1, w2, int(direction.value), distance, 1)
        current_table_index += 1
    return (t, current_table_index)

def read_corpus(fn):
    "Reads the corpus file and returns the text"
    LOG.debug("=== fn ===")
    LOG.debug(fn)

    with codecs.open(os.path.expandvars(fn), 'r', 'utf-8') as fin:
        LOG.debug("=== fn ===")
        LOG.debug(fn)
        content = fin.read()
    return content


def generate_component_ngrams(corpus_file, max_distance=3):
    """A component ngram table is like

    | c1  | c2 | direction | diacritic | distance | freq |
    |     |    |           |           |          |      |
    | ar  | b  | L         | F         | 1        | 20   |
    | b   | ar | R         | F         | 1        | 10   |
    | b   |  1 | U         | T         | 1        | 303  |


    * Rules

    A string arbu1 elxu1bu2l is divided into components
    [ar, b, u1, e, lxbl, u1, u2]

    From this list, we generate the above component ngram by processing windows
    of components.

    c1 and c2 are component labels.

    direction is L(eft), R(ight), U(nder), or O(ver). U and O are normally
    reserved for diacritics and L and R are for stems.

    distance is the $n-1$ in n-gram. For stem bigrams it is 1. The distance
    parameter determines the largest window of stems.

    """

    corpus_text = read_corpus(corpus_file)

    # TODO: We need a cleanup here for bilingual text
    visenc_corpus = aua.arabic_to_visenc(corpus_text)
    visenc_list = aua.split_visenc(visenc_corpus)
    table = _init_component_ngram_table(size=10000)
    current_table_index = 0
    for i, c1 in enumerate(visenc_list):
        if aua.is_diacritic(c1):
            continue

        stem_dist_counter = 1
        for j, c2 in enumerate(visenc_list[(i+1):]):
            if stem_dist_counter == 1 and aua.is_diacritic(c2):
                if c2[0] == 'u':
                    direction = Direction.UNDER
                elif c2[0] == 'o':
                    direction = Direction.OVER
                else:
                    LOG.error("Need u or o as the first letter"
                              "in diacritic {}".format(c2))
                    continue
                table, current_table_index = _add_component_table_record(table,
                                                                         c1,
                                                                         c2,
                                                                         direction,
                                                                         True,
                                                                         stem_dist_counter,
                                                                         current_table_index)

                # As this is only diacritic, we don't increase stem_dist_counter

            elif stem_dist_counter < max_distance and aua.is_stem(c2):
                table, current_table_index = _add_component_table_record(table,
                                                                         c1,
                                                                         c2,
                                                                         Direction.LEFT,
                                                                         False,
                                                                         stem_dist_counter,
                                                                         current_table_index)
                table, current_table_index = _add_component_table_record(table,
                                                                         c2,
                                                                         c1,
                                                                         Direction.RIGHT,
                                                                         False,
                                                                         stem_dist_counter,
                                                                         current_table_index)

                stem_dist_counter += 1
            else:
                break

    return table

def generate_word_ngrams(corpus_file, max_distance=3):
    corpus_text = read_corpus(corpus_file)

    visenc_corpus = aua.arabic_to_visenc(corpus_text)
    word_list = visenc_corpus.split()
    table = _init_word_ngram_table(size=10000)
    current_table_index = 0
    for i, w1 in enumerate(word_list):
        for j, w2 in enumerate(word_list[(i+1):]):
            if j < max_distance:
                table, current_table_index = _add_word_table_record(table,
                                                                    w1,
                                                                    w2,
                                                                    Direction.LEFT,
                                                                    j+1,
                                                                    current_table_index)
                table, current_table_index = _add_word_table_record(table,
                                                                    w2,
                                                                    w1,
                                                                    Direction.RIGHT,
                                                                    j+1,
                                                                    current_table_index)
            else:
                break
    return table

def _init_component_ngram_table(size=10000):
    table = np.zeros(dtype=[('c1', 'S64'),
                            ('c2', 'S64'),
                            ('direction', np.int),
                            ('diacritic', np.bool),
                            ('distance', np.int),
                            ('frequency', np.int)],
                     shape=(size,))
    return table
 
def _init_word_ngram_table(size=10000):
    table = np.zeros(dtype=[('w1', 'S64'),
                            ('w2', 'S64'),
                            ('direction', np.int),
                            ('distance', np.int),
                            ('frequency', np.int)],
                     shape=(size,))
    return table

def _merge_component_arrays(arr1, arr2):
    assert arr1.dtype == arr2.dtype
    insertion_index = arr1.shape[0]
    orig_index = arr1.shape[0]
    for i in range(orig_index):
        r1 = arr1[i]
        r2 = arr2[np.where(arr2['c1'] == r1['c1']
                           and arr2['c2'] == r1['c2']
                           and arr2['direction'] == r1['direction']
                           and arr2['diacritic'] == r1['diacritic']
                           and arr2['distance'] == r1['distance'])]
    if r2.size > 0:
        r1['frequency'] += r2['frequency']
    else:
        if insertion_index >= arr1.shape[0]:
            arr1 = _array_x2(arr1)
            arr1[insertion_index] = r2
            insertion_index += 1
    arr1 = arr1[:insertion_index]
    return arr1

def _merge_word_arrays(arr1, arr2):
    assert arr1.dtype == arr2.dtype
    insertion_index = arr1.shape[0]
    orig_index = arr1.shape[0]
    for i in range(orig_index):
        r1 = arr1[i]
        r2 = arr2[np.where(arr2['w1'] == r1['w1']
                           and arr2['w2'] == r1['w2']
                           and arr2['direction'] == r1['direction']
                           and arr2['distance'] == r1['distance'])]
    if r2.size > 0:
        r1['frequency'] += r2['frequency']
    else:
        if insertion_index >= arr1.shape[0]:
            arr1 = _array_x2(arr1)
            arr1[insertion_index] = r2
            insertion_index += 1
    arr1 = arr1[:insertion_index]
    return arr1



def ngrams_for_dir(the_dir, max_component_distance=1, max_word_distance=1):
    "Builds ngrams for all files in the_dir"

    files = aui.recursive_file_list(the_dir, f_regex="\.txt$")
    component_ngrams = _init_component_ngram_table(size=0)
    word_ngrams = _init_word_ngram_table(size=0)
    for f in files:
        full_f = os.path.join(the_dir, f)
        fcng = generate_component_ngrams(full_f, max_distance=max_component_distance)
        component_ngrams = _merge_component_arrays(component_ngrams, fcng)
        fwng = generate_word_ngrams(full_f, max_distance=max_word_distance)
        word_ngrams = _merge_word_arrays(word_ngrams, fwng)

    return component_ngrams, word_ngrams
