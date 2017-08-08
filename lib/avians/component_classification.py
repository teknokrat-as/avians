
import avians.util.image as aui
import avians.nn.common as anc
import cv2
import numpy as np
import os

import logging
LOG = logging.getLogger(__name__)

def component_classification(imgf, 
                             classifier_file,
                             text_detector_file,
                             class_dir,
                             classification_feature_size,
                             text_detection_feature_size,
                             output_dir='/tmp/classification-results'): 
    
    img = cv2.imread(imgf)
    imgf_base = os.path.splitext(os.path.basename(imgf))[0]
    components, stats = aui.image_segmentation(img,
                                               do_invert=True,
                                               max_area=0.2,
                                               min_area=20)
    LOG.debug("=== components.shape ===")
    LOG.debug(components.shape)
    LOG.debug("=== stats.shape ===")
    LOG.debug(stats.shape)
    LOG.debug("=== stats ===")
    LOG.debug(stats)
    aui.write_image_array(components,
                          key=imgf_base,
                          outdir="/tmp/")

    classification_components = aui.resize_components(np.array(components), 
                                                      np.array(stats),
                                                      classification_feature_size,
                                                      classification_feature_size)
    text_detection_components = aui.resize_components(np.array(components),
                                                      np.array(stats),
                                                      text_detection_feature_size,
                                                      text_detection_feature_size)
    classification_components = classification_components / 255
    text_detection_components = text_detection_components / 255

    classifier = anc.load_keras_model(os.path.expandvars(classifier_file))
    text_detector = anc.load_keras_model(os.path.expandvars(text_detector_file))
    class_names = sorted(os.listdir(class_dir))
    classification_components = classification_components.reshape(
        (classification_components.shape[0],
         1,
         classification_components.shape[1],
         classification_components.shape[2]))

    text_detection_components = text_detection_components.reshape(
        (text_detection_components.shape[0],
         1,
         text_detection_components.shape[1],
         text_detection_components.shape[2]))
    text_predictions = text_detector.predict(text_detection_components)
    class_predictions = classifier.predict(classification_components)

    for i in range(stats.shape[0]):
        # if text_predictions[i] > 0.001: 
        predicted = np.argmax(class_predictions[i])
        the_dir = os.path.join(output_dir, class_names[predicted])
        # else: 
        #     the_dir = os.path.join(output_dir, "nontext")

        component_img = classification_components[i, 0]
        x1 = stats['x'][i]
        x2 = stats['x'][i] + stats['w'][i]
        y1 = stats['y'][i]
        y2 = stats['y'][i] + stats['h'][i]
        region_img = img[x1:x2, y1:y2]

        if not os.path.isdir(the_dir): 
            os.makedirs(the_dir)

        component_file = os.path.join(the_dir, 
                                      "cmp-{}--x-{}-y-{}.png".format(
                                          imgf_base,
                                          stats['x'][i],
                                          stats['y'][i]))
        region_file = os.path.join(the_dir, 
                                   "reg-{}--x-{}-y-{}.png".format(
                                       imgf_base,
                                       stats['x'][i],
                                       stats['y'][i]))
        cv2.imwrite(component_file, component_img)
        cv2.imwrite(region_file, region_img)
