

import avians.nn.text_detection_ac_v1 as antdav1

COMPONENT_IMAGES = "/home/iesahin/Datasets/leylamecnun/components-text-amiri"
WORD_LIST = "/home/iesahin/Repository/avians/doc/arabic/test-arabic-100.txt"
FONT_LIST = ["Amiri", "Droid Sans Arabic"]

def test_incremental_update(): 
    autoencoder, decoder, encoder = antdav1.text_detection_ac_v1(WORD_LIST)
    assert autoencoder is not None
