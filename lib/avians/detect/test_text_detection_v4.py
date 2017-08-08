
from . import text_detection_v4 as tdv4
from datetime import datetime as dt
import os
import glob

def test_text_detection_v4():
    model = "/tmp/text_detection_cnn_v1-2016-06-22-02:27:40/model--2016-06-22-02:50:04.h5"

# os.path.expandvars("$HOME/Annex/Arabic/text_detection_cnn_v1-2016-06-20-17:26:10/model--2016-06-21-14:20:39.h5")
    imgs = glob.glob("/home/iesahin/Annex/Arabic/working-set-1/*.jpg")
    print(imgs)
    for i, img in enumerate(imgs):
        output = "/tmp/text-detection-output-{}.jpg".format(i)
        tdv4.main_caller(model, img, output)
        assert os.path.exists(output)
