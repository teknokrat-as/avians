
import numpy as np
import cv2


import logging
LOG = logging.getLogger(__name__)

def chain_code(contour):
    """This produces a chain code from the contour in the form

    1 4 7
    2 c 8
    3 6 9

    So a string like

    ***
       *
    
    is 

    889
"""
    cx = contour[:, 0, 0]
    cy = contour[:, 0, 1]
    dcx = np.roll(cx, 1) - cx
    dcy = np.roll(cy, 1) - cy
    return (3 * dcx + dcy) + 5


def chain_code_from_img(binary_img): 

    # LOG.debug("=== binary_img.dtype ===")
    # LOG.debug(binary_img.dtype)
    
    ii, contours, hier = cv2.findContours(binary_img,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)

    cnt = contours[0]

    cnt[:, 0, 0] = cnt[:, 0, 0] - cnt[:, 0, 0].min()
    cnt[:, 0, 1] = cnt[:, 0, 1] - cnt[:, 0, 1].min()
    min_index = np.argmin(cnt[:, 0, 0] + cnt[:, 0, 1])
    cnt[:, 0, 0] = np.roll(cnt[:, 0, 0], min_index)
    cnt[:, 0, 1] = np.roll(cnt[:, 0, 1], min_index)
    return chain_code(cnt)
