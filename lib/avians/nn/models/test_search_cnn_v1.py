

import avians.nn.models.search_cnn_v1 as anmscv1
import avians.nn.common as anc

def test_train_model_from_dataset():
    classifier, feature_extractor = anmscv1.train_model_from_dataset(dataset_dir="$HOME/Annex/Arabic/component-dataset-170K",
                                                                     early_stop_patience=500)
    anc.save_and_upload_model(classifier, "/home/iesahin/Annex/Arabic/", "letter_group_classifier")
    anc.save_and_upload_model(feature_extractor, "/home/iesahin/Annex/Arabic/", "letter_group_feature_extractor")
