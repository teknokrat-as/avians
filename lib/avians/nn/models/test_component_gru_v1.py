import avians.nn.models.component_gru_v1 as anmcgv1
import avians.nn.common as anc

def test_train_model_from_dataset():
    classifier = anmcgv1.train_model(dataset_dir=
                                     "/home/iesahin/Annex/Arabic/arabic-component-dataset-518")
    anc.save_and_upload_model(classifier,
                              "/home/iesahin/Annex/Arabic/",
                              "component_gru_v1")
