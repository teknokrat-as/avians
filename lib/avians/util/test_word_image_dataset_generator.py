
import avians.util.word_image_dataset_generator as widg
import numpy.random as nr
import os

def test_letter_group_image_dataset_generator(): 
    outdir = "/tmp/test-dataset-generation-{}/".format(nr.randint(10000))
    widg.letter_group_image_dataset_generator(n_images=100, 
                                              output_dir=outdir)
    assert False
    assert os.path.exists(outdir)
