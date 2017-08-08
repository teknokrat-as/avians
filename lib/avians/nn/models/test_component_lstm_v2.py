
import avians.nn.models.component_lstm_v2 as anmclv2
import avians.nn.common as anc

def test_train_model_from_dataset():
    classifier = anmclv2.train_model(dataset_dir = "/home/iesahin/Annex/Arabic/arabic-component-dataset-518")
    anc.save_and_upload_model(classifier,
                              "/home/iesahin/Annex/Arabic/", 
                              "component_lstm_v2")
    
