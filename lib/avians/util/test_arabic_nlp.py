
import avians.util.arabic_nlp as auan
import numpy as np
import os

TEST_CORPUS_DIR = "$HOME/Annex/Arabic/cnn-arabic-utf8/business/"
TEST_CORPUS_FILE = TEST_CORPUS_DIR + np.random.choice(os.listdir(os.path.expandvars(TEST_CORPUS_DIR)))

def test_ngrams_for_dir():
    c_ngrams, w_ngrams = auan.ngrams_for_dir(TEST_CORPUS_DIR)
    assert c_ngrams.shape[0] >= w_ngrams.shape[0]

def test_generate_component_ngrams():
    MAX_DIST=3
    c_ngrams = auan.generate_component_ngrams(TEST_CORPUS_FILE, max_distance=MAX_DIST)
    assert 'c1' in c_ngrams.dtype.names 
    assert 'c2' in c_ngrams.dtype.names 
    assert 'distance' in c_ngrams.dtype.names 
    assert 'diacritic' in c_ngrams.dtype.names 
    assert 'direction' in c_ngrams.dtype.names 
    assert c_ngrams['distance'].max() <= MAX_DIST

def test_generate_word_ngrams(): 
    MAX_DIST=3
    w_ngrams = auan.generate_word_ngrams(TEST_CORPUS_FILE, max_distance=MAX_DIST)
    assert 'w1' in w_ngrams.dtype.names 
    assert 'w2' in w_ngrams.dtype.names 
    assert 'distance' in w_ngrams.dtype.names 
    assert 'direction' in w_ngrams.dtype.names 
    assert w_ngrams['distance'].max() <= MAX_DIST


