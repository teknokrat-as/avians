

import avians.util.arabic as aua
from nose.tools import with_setup
import logging
LOG = logging.getLogger(__name__)

def test_split_visenc(): 
    assert aua.split_visenc("ellh") == ["e", "llh"]
    assert aua.split_visenc("emrh") == ["e", "mr", "h"]
    assert aua.split_visenc("bu1sm||ellh||elrxmebo1||elrxbu2m") == ["bu1sm||", "e", "llh||", 
                                                                    "e", "lr", "xme", "bo1||",
                                                                    "e", "lr", "xbu2m"]

def test_delete_visenc_stems(): 
    assert aua.delete_visenc_stems("ao1bu1") == "o1u1"
    assert aua.delete_visenc_stems("bo1xo1lie") == "o1o1i"

def test_system_arabic_fonts(): 
    fl = aua.system_arabic_fonts()
    assert len(fl) > 0
