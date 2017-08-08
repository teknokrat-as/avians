
import os
import video_ocr as ocr
import avians.util as au
import avians.util.image as aui

def test_load_model():
    model_file = os.path.expandvars("$HOME/Annex/Arabic/component_ctc_v5.h5")
    
    m = ocr.load_model(model_file)
    
    assert m != None
   
   
def test_shard_video(): 
    ## FIXME: Change the following path to a video in Annex after it's configured
    video_file = os.path.expandvars("$HOME/Downloads/test_video_3.mjpg")
    random_dir1 = au.get_random_dir("/dev/shm")
    ocr.shard_video(video_file,
                    random_dir1,
                    1)
    out_list1 = aui.file_list(random_dir1)
    assert len(out_list1) > 0
    random_dir2 = au.get_random_dir("/dev/shm")
    ocr.shard_video(video_file,
                    random_dir2,
                    4)
    out_list2 = aui.file_list(random_dir2)
    assert len(out_list1)*4 <= len(out_list2)
    
def test_detect_text_regions(): 
    text_image = write_text_to_image("hello",
                                      1024, 768)
    out_file = au.get_random_file("/dev/shm", ".png")
    cv2.imwrite(out_file,
                text_image)
                       
    regs = ocr.detect_text_regions(out_file)
    assert regs.shape == (1, text_image.shape[0], text_image.shape[1])
    
    ### TODO: Add another test for a real image from the dataset
    
def test_shard_image():
    text_image = write_text_to_image("h e l l o",
                                      1024, 768)
    sharded = ocr.shard_image(text_image)
    assert sharded.shape == (5, 1024, 768)
         
    
def test_prepare_for_detection(): 
    text_image = write_text_to_image("h e l l o",
                                      1024, 768)
    sharded = ocr.shard_image(text_image)

    resized = ocr.prepare_for_text_detection(sharded,
                                             512,
                                             64,
                                             5)
    assert resized.shape == (5 * 5, 512, 64)

    pass
    

def test_ocr_func():
    text = ocr.ocr_func("/home/iesahin/Annex/Arabic/test-1.png",
                        models=['/home/iesahin/tmp/res-634/lm-26-wobmodel-component_lstm_v3-2016-09-23-14:34:55.h5'],
                        datasets=['/home/iesahin/tmp/res-634/lm-26-wob/component_lstm_v3.npz'])
    assert text == """داستان لیلی و مجنون
بسمالله الرحمن الرحیم
رباعی
ای نشأه حسنی عشقه تأثیر قیلن
عشقیله بناء كونی تعمیر قیلن
لیلی سر زلفنی گره گیر قیلن
مجنون حزین بویننه زنجیر قیلن
"""
