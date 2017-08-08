
import train_rnn as tr
import generate_dataset as gd
from nose.tools import with_setup
import os

import logging
LOG = logging.getLogger(__name__)

POSITIVE_SET = os.path.expandvars("/tmp/text-detection-images/")
NEGATIVE_SET = os.path.expandvars("$HOME/Annex/Arabic/working-set-1-non-text")
POSITIVE_WORDS = os.path.expandvars("$HOME/Repository/avians/doc/arabic/arabic-wordlist-65k.txt")
    
def test_generate_dataset_words(): 
    if os.path.isdir(NEGATIVE_SET): 
        negative_samples = [os.path.join(NEGATIVE_SET, f) for f in os.listdir(NEGATIVE_SET)]
    else: 
        negative_samples = [NEGATIVE_SET]

    dataset = tr.generate_dataset_by_words(POSITIVE_WORDS, negative_samples)
    assert len(dataset) > 0 is not None

def test_generate_dataset(): 
    if os.path.isdir(POSITIVE_SET): 
        positive_samples = [os.path.join(POSITIVE_SET, f) for f in os.listdir(POSITIVE_SET)]
    else: 
        positive_samples = [POSITIVE_SET]
    
    if os.path.isdir(NEGATIVE_SET): 
        negative_samples = [os.path.join(NEGATIVE_SET, f) for f in os.listdir(NEGATIVE_SET)]
    else: 
        negative_samples = [NEGATIVE_SET]

    dataset = tr.generate_dataset(positive_samples, negative_samples)
    assert dataset is not None
    
# def test_train_rnn(): 
#     tr.train_rnn(POSITIVE_DIR, NEGATIVE_DIR, "/tmp/text-detection-model.h5")
    
def test_train_cnn(): 
    tr.train_cnn(POSITIVE_DIR, NEGATIVE_DIR, "/tmp/text-detection-model.h5")



