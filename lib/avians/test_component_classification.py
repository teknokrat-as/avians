
import avians.component_classification as acc
import avians.util.image as aui
import os
import logging
LOG = logging.getLogger(__name__)


def test_component_classification():
    class_model = "/home/iesahin/Annex/Arabic/model-letter_group_classifier-2016-07-09-17:04:47.h5"
    td_model = "$HOME/Annex/Arabic/text-detection-model-acc-0.91-single-channel-large-dataset/text-detection-model-2016-06-15-21:26:40.h5"
    dataset_dir = "/home/iesahin/Annex/Arabic/arabic-component-dataset-518"

    input_dir = "$HOME/Annex/Arabic/syrianacc"
    outdir = os.path.join('/tmp', os.path.basename(input_dir))
    image_files = aui.file_list(input_dir)
    LOG.debug("=== image_files ===")
    LOG.debug(image_files)
    for f in image_files: 
        acc.component_classification(
            f,
            class_model,
            td_model,
            dataset_dir,
            classification_feature_size=64,
            text_detection_feature_size=32,
            output_dir=outdir)

    assert False

