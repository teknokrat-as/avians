import numpy as np
import cv2


import logging
LOG = logging.getLogger(__name__)

def projection_from_img(img): 
    nz = np.vstack(img.nonzero())
    proj_len = img.shape[1]
    proj_arr = np.zeros((3, proj_len), np.int)
    for i in range(proj_len): 
        nzi = nz[0, nz[1] == i]
        if len(nzi) > 0: 
            proj_arr[0, i] = nzi.max()
            proj_arr[1, i] = nzi.min()
            proj_arr[2, i] = len(nzi)
    return proj_arr


