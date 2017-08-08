
import avians.search as avs
import os

def test_search_word_in_library(): 
    word = "حلب"
    library = os.path.expandvars("$HOME/tmp/")
    # library = os.path.expandvars("$HOME/Annex/Arabic/working-set-1/")

    search_results = avs.search_word_in_library(word, library)
    assert len(search_results) > 0, "{}".format(search_results)

