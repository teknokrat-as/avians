
from . import text_detection_v3 as tdv3
from datetime import datetime as dt
import os

def test_text_detection_v2():
    autoencoder = os.path.expandvars("$HOME/Annex/Arabic/text_detection_ac_v1-2016-06-10-04:02:26/model-autoencoder-2016-06-11-13:04:06.h5")
    img = os.path.expandvars("$HOME/Annex/Arabic/working-set-1/rt-1-img8094.jpg")
    output = "/tmp/text-detection-output-{}.jpg".format(dt.now().strftime("%T"))
    tdv3.main_caller(autoencoder, img, output)

    assert os.path.exists(output)
