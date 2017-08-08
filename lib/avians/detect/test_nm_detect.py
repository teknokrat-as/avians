
import os
import cv2
import numpy as np
from nose.tools import with_setup
import avians.detect.nm_detect as adnd
from datetime import datetime
import logging
LOG = logging.getLogger(__name__)

test_img_path = os.path.expandvars("$HOME/Annex/Arabic/working-set-1/rt-1-img8094.jpg")

def test_find_channels():
    test_img = cv2.imread(test_img_path)
    print("test_find_channels: ", adnd.find_channels(test_img))

def test_find_text_points():
    test_img = cv2.imread(test_img_path)
    channels = adnd.find_channels(test_img)
    tp = adnd.find_text_regions(test_img, channels)
    LOG.debug("=== tp ===")
    LOG.debug(tp)
    assert False

def test_find_text_components(): 
    test_img = cv2.imread(test_img_path)
    all_objects, all_stats = adnd.find_text_components(test_img)
    output_folder = '/tmp/text_detection_' + datetime.now().strftime('%s')
    os.makedirs(output_folder)
    for i in range(all_objects.shape[0]):
        cv2.imwrite('{}/{}.png'.format(output_folder,
                                       i),
                    all_objects[i])
    assert all_objects.shape[0] == all_stats.shape[0]

def test_main_inner():
    assert os.path.exists(test_img_path)
    rn = np.random.randint(10000)
    adnd.main_inner(test_img_path,
                    1,
                    '/tmp/test_nm_{}/'.format(rn),
                    '/tmp/test_nm_cnt_{}/'.format(rn))
    
