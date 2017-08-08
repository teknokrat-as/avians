
import avians.detect.text_detection as adtd
from nose.tools import with_setup
import os

TEST_IMAGE_NAME = os.path.expandvars("$HOME/Annex/Arabic/working-set-1/rt-1-img8094.jpg")

# (pyvenv-activate "/home/iesahin/Environments/dervaze")

def test_detect_text():
    adtd.detect_text(TEST_IMAGE_NAME, 2, 5, False, False)

def test_detect_text_via_components(): 
    adtd.detect_text_via_components(TEST_IMAGE_NAME, 3)
    assert False
