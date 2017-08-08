
import avians.ocr as ocr

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
