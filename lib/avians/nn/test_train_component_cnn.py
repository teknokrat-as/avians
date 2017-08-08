
import avians.nn.train_component_cnn as antcc

def test_component_cnn(): 
    # model = antcc.train_component_cnn("$HOME/Annex/Datasets/leylamecnun/components-amiri-74K-1200-dpi")
    model = antcc.train_component_cnn("/tmp/train_component_cnn-2016-07-06-04:52:12/dataset.npz")
    assert model
