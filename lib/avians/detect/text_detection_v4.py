
import avians.util.image as aui
import avians.nn.common as anc
from datetime import datetime as dt

import argparse
import cv2
import numpy as np

import logging
LOG = logging.getLogger(__name__)

def text_detection_v4(image, 
                      model, 
                      prob_threshold=0.01, 
                      size=32,
                      strides=16): 

    if len(image.shape) > 2: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_arr = aui.split_image(image, size, size, strides, strides)
    image_arr = image_arr.swapaxes(1, 3).swapaxes(2, 3)
    feature_arr = (image_arr - 128) / 128
    predictions = model.predict(feature_arr, batch_size=128, verbose=1)
    td_image = np.zeros(shape=image.shape, dtype=np.float)
    n_col = np.ceil(td_image.shape[1] / strides)
    for i, p in enumerate(predictions):
        x1 = (i % n_col) * strides
        x2 = x1 + size
        y1 = (i // n_col) * strides
        y2 = y1 + size
        td_image[y1:y2, x1:x2] = p

    outimage = np.ceil(td_image * image)
    return outimage

def load_model(fn):
    m = anc.load_keras_model(fn,
                             loss="binary_crossentropy",
                             optimizer="adadelta")
    m.summary()
    return m

def main_caller(model_f,
                image,
                output): 
    
    img = cv2.imread(image, 0)
    model = load_model(model_f)
    result = text_detection_v4(img, model)
    cv2.imwrite(output, result)
    print("Result Saved To: {}".format(output))

def main(): 
    parser = argparse.ArgumentParser(description="""Detects textual regions on
    the given image using the supplied model generated by train_rnn """,
                                    formatter_class=argparse.RawDescriptionHelpFormatter) # noqa
    
    parser.add_argument('model', 
                        help="Keras model name to detect text")
    parser.add_argument('image', 
                        help="Image that we are going to detect the text")
    parser.add_argument('--output', 
                        help="File to save the image", 
                        default="text-detection-result-{}.png".format(
                            dt.now().strftime("%F-%T")))

    args = vars(parser.parse_args())
    print(args)
    main_caller(**args)

if __name__ == '__main__': 
    main()