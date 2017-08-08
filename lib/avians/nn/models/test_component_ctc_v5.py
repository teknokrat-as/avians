import avians.nn.models.component_ctc_v5 as anmccv5
import avians.nn.models.component_ctc_v4 as anmccv4
import avians.nn.common as anc
import avians.util.image as aui
import numpy as np
import os
import cv2
import re
import datetime as dt

# def test_AviansModel():
#     # dataset_dir = "/home/iesahin/Archive/Annex/Arabic/arabic-component-dataset-test-small"
#     dataset_dir = "/home/iesahin/Annex/Arabic/arabic-component-dataset-518/ad/"
#     weights_file = "/home/iesahin/Repository/avians/image_ocr/Monday, 24. October 2016 12.53PM/weights03.h5"
#     file_list = aui.recursive_file_list(dataset_dir)
#     # file_list = np.random.shuffle(file_list)
#     # file_list = file_list[:1000]
#     model_obj = anmccv5.ComponentModelCTC_v5(weights_file)
#     labels = model_obj.classify(file_list)
#     print(labels)
#     for l, f in zip(labels, file_list):
#         print(l, f)
        

np.random.seed()

weights_file = "/home/iesahin/Repository/avians/image_ocr/Monday, 24. October 2016 12.53PM/weights03.h5"
dataset_dir = os.path.expandvars("$HOME/Annex/Arabic/Avians/arabic-component-dataset-518/")
annex_dir = os.path.expandvars("$HOME/Annex/Arabic/")
artifact_dir = os.path.join(annex_dir, "ctc-v5-{}".format(dt.datetime.now().strftime("%F")))

def test_init():
    # Is it able to create a model
    model = anmccv5.ComponentModelCTC_v5(weights_file)
    assert model is not None

def test_queue():
    # Change this to selectively run the tests with a single nose call
    # test_train_and_compare()
    test_classify_image_with_pretrained()
    test_train_for_a_single_word()

def check_cls(correct_cls, cls): 
    assert cls == correct_cls, "Correct: {} Returned: {}".format(correct_cls, cls)

def check_cls_among(correct_cls, cls): 
    assert correct_cls in [c for c, p in cls], "Correct: {} Returned: {}".format(correct_cls, cls) 

def test_classify_image_with_pretrained(): 
    model = anmccv5.ComponentModelCTC_v5(weights_file)
    image_files_dir = os.path.expandvars("$HOME/Annex/Arabic/Avians/arabic-component-dataset-4170/")
    image_files = aui.recursive_file_list(image_files_dir)
    image_files = np.random.choice(image_files, 200)
    assert len(image_files) > 0
    for image_file in image_files:
        assert os.path.exists(image_file)
        img = cv2.imread(image_file, 0)
        cls = model.classify_image(img)
        mm = re.match(image_files_dir + r'([a-z]+)/cc.*png', image_file)
        correct_cls = mm.group(1)
        yield check_cls, correct_cls, cls

def test_train_for_a_single_word():
    model = anmccv5.ComponentModelCTC_v5(None)
    model.train_model(dataset_dir)
    save_file = "/tmp/small-classifier.h5"
    model.save_weights(save_file)
    model_reloaded = anmccv5.ComponentModelCTC_v5(save_file)
    image_files_dir = os.path.expandvars("$HOME/Annex/Arabic/arabic-component-dataset-test-small/")
    image_files = aui.recursive_file_list(image_files_dir)
    assert len(image_files) > 0
    for image_file in image_files:
        assert os.path.exists(image_file)
        img = cv2.imread(image_file, 0)
        # cls = model.classify_image(img)
        cls_reloaded = model_reloaded.classify_image(img)
        mm = re.match(image_files_dir + r'([a-z]+)/cc.*png', image_file)
        correct_cls = mm.group(1)
        yield check_cls, correct_cls, cls_reloaded

def test_train_and_classify(): 
    current_weights_file = artifact_dir + "/weights.h5"
    if os.path.exists(current_weights_file): 
        model = anmccv5.ComponentModelCTC_v5(current_weights_file)
    else: 
        model = anmccv5.ComponentModelCTC_v5(None)
        model.train_model(dataset_dir=dataset_dir,
                          nb_epoch=100)
        if not os.path.exists(artifact_dir): 
            os.mkdir(artifact_dir)
        model.save_weights(current_weights_file)

    image_files_dir = dataset_dir
    image_files = aui.recursive_file_list(image_files_dir)
    image_files = np.random.choice(image_files, 200)
    assert len(image_files) > 0
    for image_file in image_files:
        assert os.path.exists(image_file)
        img = cv2.imread(image_file, 0)
        cls = model.classify_image(img)
        mm = re.match(image_files_dir + r'([a-z]+)/.*png', image_file)
        correct_cls = mm.group(1)
        yield check_cls_among, correct_cls, cls

def test_train_and_compare():
    weights_file = artifact_dir + "/weights.h5"
    print(weights_file)
    model_v4, test_func_v4 = anmccv4.train_model(dataset_dir)
    try: 
        os.makedirs(artifact_dir)
    except:
        print("Error creating artifact_dir")

    model_v4.save_weights(weights_file)
    model_v5 = anmccv5.ComponentModelCTC_v5(weights_file)
    test_func_v5 = model_v5._predictor
    size = 32
    tests = 20
    for i in range(tests): 
        X_data = np.random.ranf([size, 1, model_v5.img_h, model_v5.img_w])
        v4_out = test_func_v4([X_data])
        v5_out = test_func_v5([X_data])
        assert len(v4_out) == len(v5_out), "Output lengths differ: \nv4: {}\nv5: {}".format(len(v4_out), len(v5_out))
        print("V4:\n{}".format(v4_out))
        print("V5:\n{}".format(v5_out))
        for i, v4v in enumerate(v4_out):
            v5v = v5_out[i]
            assert np.all(v4v == v5v)

