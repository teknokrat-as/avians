

import avians.nn.models.component_cnn_v2 as anmccv2
import avians.nn.common as anc

def test_train_model_from_dataset():
    classifier, feature_extractor = anmccv2.train_model_from_dataset(dataset_dir="/home/iesahin/Annex/Arabic/arabic-letter-classes-240K",
                                                                     early_stop_patience=1000)
    anc.save_and_upload_model(classifier, "/home/iesahin/Annex/Arabic/", "length_classifier")
    anc.save_and_upload_model(feature_extractor, "/home/iesahin/Annex/Arabic/", "length_feature_extractor")
