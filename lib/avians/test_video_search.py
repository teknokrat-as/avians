
# import theano
# theano.config.exception_verbosity = "high"
# theano.config.optimizer = "None"
import os
import avians.video_search as vs
import avians.util.arabic as aua
import avians.util.image as aui
import avians.util as au
import os
import numpy as np
from glob import glob as ls
import logging
LOG = logging.getLogger(__name__)
from nose.tools import timed

def check_equal(a, b):
    assert a == b

def passing_test_array_2x():
    t = np.random.rand(10, 10)
    t2 = vs._array_x2(t)
    assert t[0, 0] == t2[0, 0]
    assert np.all(t[:, 0] == t2[:10, 0])
    assert np.all(t == t2[:10, :])
    assert np.all(t2[10:, :] == 0)

def test_labelmap_to_dir():
    lmfs = ls(os.path.expandvars('$HOME/Annex/Arabic/working-set-1/*/labelmap.lm2'))
    assert len(lmfs) > 0, str(lmfs)
    for lmf in lmfs:
        lmdir = '/tmp/{}/'.format(os.path.basename(lmf))
        vs.labelmap_to_dir(lmf, lmdir)
        assert os.path.exists(lmdir)
        assert os.path.exists(os.path.join(lmdir, 'labels.txt'))

@timed(600)
def test_search_string_in_component_map():
    lmf = os.path.expandvars('$HOME/Annex/Arabic/working-set-1/rt-1-img8094/labelmap.lm2')
    LOG.debug(lmf)
    lm = vs.load_labelmap(lmf)
    LOG.debug(lm)
    res = vs.search_string_in_component_map(lm, 'llr')
    LOG.debug("=== search result ===")
    LOG.debug(res)
    assert res.shape[0] > 0

def passing_test_image_to_labelmap():
    image_files = aui.recursive_file_list('$HOME/Annex/Arabic/working-set-1')
    for f in image_files:
        labelmap_file, imgs, stats, labels, cmap = vs.image_to_labelmap(f)

        assert labelmap_file.endswith("lm2")
        assert os.path.exists(labelmap_file)

def test_load_labelmap():
    labelmap_files = ls(os.path.expandvars('$HOME/Annex/Arabic/working-set-1/*/labelmap.lm2'))
    assert len(labelmap_files) > 0
    for f in labelmap_files:
        lm = vs.load_labelmap(f)
        imgs = lm['images']
        stats = lm['stats']
        labels = lm['labels']
        if imgs.ndim > 0:
            assert imgs.shape[0] == stats.shape[0]
            assert labels.shape[0] == imgs.shape[0]

def test_component_graph():
    import matplotlib.pyplot as plt
    import networkx as nx
    the_dir = os.path.expandvars('$HOME/Annex/Arabic/working-set-1/')
    lmf_files = aui.file_list(the_dir, f_regex='.*lm2$')
    for lmf in lmf_files:
        lm = vs.load_labelmap(lmf)
        lm_labels = lm['labels']
        LOG.debug("=== lm_labels.dtype ===")
        LOG.debug(lm_labels.dtype)
        LOG.debug("=== lm_labels.shape ===")
        LOG.debug(lm_labels.shape)
        LOG.debug("=== lm_labels.size ===")
        LOG.debug(lm_labels.size)
        if lm_labels.size > 0:
            max_labels = lm_labels[:, 0]['label']
            LOG.debug("=== max_labels.shape ===")
            LOG.debug(max_labels.shape)
            LOG.debug("=== max_labels.dtype ===")
            LOG.debug(max_labels.dtype)
            LOG.debug("=== max_labels ===")
            LOG.debug(max_labels)
            cmap = aui.generate_component_map(lm['stats'], PROXIMITY_BOUND=1)
            graph = aui.component_map_to_nx(lm['stats'], cmap)
            pos = nx.spring_layout(graph)
            LOG.debug("=== pos ===")
            LOG.debug(pos)
            nx.draw_networkx_nodes(graph, pos,
                                   node_size=1000)
            labels = {i: max_labels[i].decode() for i in range(max_labels.shape[0])}
            nx.draw_networkx_labels(graph, pos, labels)
            dists = nx.get_edge_attributes(graph, 'distance')
            ## TODO: Draw distance and angle to the graph
            nx.draw_networkx_edges(graph, pos)
            nx.draw_networkx_edge_labels(graph, pos, dists)
            plt.savefig("{}.png".format(lmf), dpi=2000)
            plt.clf()
            assert graph.number_of_nodes() == len(labels)

def test_search_string_in_dir():
    result_len = 100
    results = vs.search_string_in_dir('$HOME/Annex/Arabic/working-set-1',
                                      'xlbu1',
                                      result_len)
    assert results.shape[0] > 0
    assert results.shape[0] <= result_len

# def test_isolate_largest_components():
#     word = aua.visenc_to_arabic('emrh')  # Emre
#     target_dir = au.get_random_dir('/dev/shm')
#     word_imgs = vs.generate_word_images(word, None, target_dir)
#     assert len(word_imgs) > 0
#     segmented_word_imgs = [aui.text_image_segmentation(wi) for wi in word_imgs]
#     largest_ci_list = vs.isolate_largest_components(segmented_word_imgs)
#     assert len(largest_ci_list) == len(segmented_word_imgs)
#     for largest_ci, swi in zip(largest_ci_list, segmented_word_imgs):
#         # swi_imgs = swi[0]
#         # swi_stats = swi[1]
#         largest_area = swi[1][largest_ci]['area']
#         assert np.all(largest_area >= swi[1]['area'])

# def test_generate_word_images():
#     word = aua.visenc_to_arabic('xlbu1') # Aleppo
#     target_dir = "/dev/shm/test_generate_word_imgs/"
#     imgs = vs.generate_word_images(word, None, target_dir)
#     assert len(imgs) > 0
#     files = os.listdir(target_dir)
#     assert len(files) >= len(imgs)

# def test_word_via_custom_model():
#     word = aua.visenc_to_arabic('xlb') # Searching Aleppo
#     image_lib_dir = os.path.expandvars('$HOME/Annex/Arabic/working-set-1')
#     assert os.path.exists(image_lib_dir)
#     results = vs.search_word_via_custom_model(word, image_lib_dir)
#     ## TODO: ADD more sensible testing condition
#     assert len(results) > 0
