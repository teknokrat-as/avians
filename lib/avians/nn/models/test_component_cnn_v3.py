
import avians.nn.models.component_cnn_v3 as anmccv3
import avians.nn.common as anc

def test_train_model_from_dataset():
    classifier = anmccv3.train_model_from_dataset(dataset_dir="/home/iesahin/Annex/Arabic/arabic-component-dataset-4170",
                                                                     early_stop_patience=0)
    anc.save_and_upload_model(classifier, "/home/iesahin/Annex/Arabic/", "component_classifier_4170_v3")

