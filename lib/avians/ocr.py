

import avians.nn.common as anc
import os

def single_model_ocr(input_file,
                     model_file): 
    
    ## TODO write a function to return text from models and input
    pass
                                           


def prepare_text_images(image_arr,
                        stats_arr, 
                        avians_model):
    """Prepare component images so that they are suitable to X arrays in datasets.

    :param image_arr: A set of images which contain white on black components

    :param stats_arr: Shows the location and other stats of components in image_arr
    
    :param model_module: The module to recreate the model and preprocess the images

    """

    n_images, w_img, h_img = image_arr.shape
    X = dataset['X']
    ds_n, ds_height, ds_width = X.shape
    x1 = stats_arr['x']
    x2 = stats_arr['x'] + stats_arr['w']
    y1 = stats_arr['y']
    y2 = stats_arr['y'] + stats_arr['h']
    c_arr = np.zeros(shape=(n_images,
                            ds_height,
                            ds_width))

    for i_img in range(n_images):
        img = image_arr[i]
        s = stats_arr[i]
        cp_img = img[x1[i]:x2[i], y1[i]:y2[i]].copy()
        r_img = cv2.resize(cp_img,
                           dsize=(cp_img.shape[1], 
                                  ds_height))
        c_arr[i, :, r_img.shape[1]] = r_img
    return c_arr

def ocr_func(input_image_file,
             models=["$HOME/Annex/Arabic/default_model.h5"],
             datasets=["$HOME/Annex/Arabic/default_dataset.npz"]):
    """Entry point for the OCR functionality. 

    :param input_image_file: Filename to convert to text. The file should
    contain a white on black binary image.

    :param models: A list of filenames to load models 

    :param datasets: A list of filenames to load datasets

    """
    models = anc.load_models(**{str(i): os.path.expandvars(f) 
                                for i, f in enumerate(models)})
    datasets = anc.load_datasets(**{str(i): os.path.expandvars(f) 
                                    for i, f in enumerate(datasets)})
    image_arr, stats_arr = anc.text_image_segmentation(input_image_file)
    component_arrs = prepare_text_images(image_arr,
                                         stats_arr,
                                         datasets)

    labeled_components = label_components(image_arr,
                                          stats_arr,
                                          models,
                                          datasets)
    text = produce_text(labeled_components)

    return text
