
import os
import avians.nn.text_detection_cnn_v1 as avtdcv1

ARABIC_WORDLIST = os.path.expandvars("$HOME/Repository/avians/doc/arabic/arabic-wordlist-65k.txt")
NONTEXT_DIR = os.path.expandvars("$HOME/Annex/Arabic/various-non-text")

def test_train_cnn_by_words(): 
    model = avtdcv1.train_cnn_by_words(ARABIC_WORDLIST, 
                                       NONTEXT_DIR,
                                       "/tmp/td-model")
    assert model
