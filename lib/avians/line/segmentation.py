
import numpy as np
import cv2
import harfbin.util.image_utils as huiu

def find_lines_by_dilate(img, dilate_factor=0.2): 
    """Finds the lines by finding components, and using dilate to 
    expand boundaries."""
    component_arr = huiu.text_image_segmentation(img)
    dilated_components = dilate_array(component_arr, factor=dilate_factor)
    merged_image = merge_image_array(dilated_components)
    connected_regions = huiu.text_image_segmentation(merged_image)
    min_rectangles = minimum_box_points(connected_regions)
    line_masks = draw_min_rectangles(min_rectangles,
                                     img_shape=img.shape)
    line_images = mask_image(img, line_masks)
    return line_images

def find_lines_by_box(img, box_enlargement=0.3): 
    component_arr = huiu.text_image_segmentation(img)
    component_rects = minimum_rectangles(component_arr)
    enlarged_rects = boxify(resize_rectangles(component_rects, box_enlargement))
    rectangle_imgs = draw_min_rectangles(enlarged_rects, img_shape=img.shape)
    merged_image = merge_image_array(rectangle_imgs)
    connected_rects = huiu.text_image_segmentation(merged_image)
    line_rects = minimum_box_points(connected_rects)
    line_masks = draw_min_rectangles(line_rects, 
                                     img_shape=img.shape)
    line_images = mask_image(img, line_masks)
    return line_images

def dilate_array(img_arr, factor):
    "Dilate each image in the array"
    dilated_arr = np.zeros_like(img_arr)
    bounds = huiu.component_boundaries(img_arr)
    wh = np.vstack((bounds['w'], bounds['h']))
    max_wh = np.apply_over_axes(np.max, wh, 0)[0]
    wh2 = np.sqrt(wh[0] * wh[0] + wh[1] + wh[1])
    dilate_counts = np.int0(np.floor(wh2 * factor))
    n_img = img_arr.shape[0]
    for i in range(n_img): 
        dilated_arr[i] = huiu.dilate(img_arr[i], dilate_counts[i])
    return dilated_arr

def resize_rectangles(rect_list, factor): 
    return [(p, (s[0]*factor, s[1]*factor), a) for (p, s, a) in rect_list]

def boxify(rect_list): 
    return np.array([np.int0(cv2.boxPoints(r)) for r in rect_list])

# def enlarge_rectangles(rect_arr, factor): 
#     x1 = rect_arr[0]
#     y1 = rect_arr[1]
#     x2 = rect_arr[2]
#     y2 = rect_arr[3]
#     assert np.all(x1 > x2)
#     assert np.all(y1 > y2)
#     alpha = rect_arr[4]
#     dx = x1 - x2
#     dy = y1 - y2
#     width = np.sqrt(dx * dx + dy * dy)
#     expansion = width * factor / 2 
#     new_x1 = x1 - expansion * np.cos(alpha)
#     new_x2 = x2 + expansion * np.cos(alpha)
#     new_y1 = y1 - expansion * np.sin(alpha)
#     new_y2 = y2 + expansion * np.sin(alpha)
#     new_rect_array = np.vstack((new_x1, new_y1, new_x2, new_y2, alpha))
#     return new_rect_array

def minimum_rectangles(img_arr): 
    "Find minimum rectangles around components in the image array"
    n_img = img_arr.shape[0]
    rect_list = [cv2.minAreaRect(
                    cv2.findNonZero(img_arr[i]))
        for i in range(n_img)
        if np.any(img_arr[i] > 0)]

    return rect_list

def minimum_box_points(img_arr): 
    "Find minimum box points around components in the image array"
    n_img = img_arr.shape[0]
    rect_list = [
        np.int0(
            cv2.boxPoints(
                cv2.minAreaRect(
                    cv2.findNonZero(img_arr[i]))))
        for i in range(n_img)
        if np.any(img_arr[i] > 0)]

    return np.array(rect_list)
    
def draw_min_rectangles(rect_arr, img_shape, fill_value=1): 
    "Draw rectangles in the rect arr to a blank image and return"
    "        rr = cv2.minAreaRect(cv2.findNonZero(t_arr[i]))\n",
    "        box = cv2.boxPoints(rr)\n",
    "        box = np.int0(box)\n",
    "        cv2.drawContours(e2, [box], 0, (255, 255, 255), 2)\n",

    n_rect = rect_arr.shape[0]
    imgs = np.zeros(shape=(n_rect, 
                           img_shape[0], 
                           img_shape[1]), dtype=np.uint8)

    for i in range(n_rect): 
        cv2.drawContours(imgs[i], [rect_arr[i]], 0, 255, thickness=-1)

    return imgs


def mask_image(img, masks): 
    """Return an image_arr each masked by masks

    :param image: An array with shape (r, c)
    :param masks: An array with shape (n, r, c)

    :rtype: An array with shape (n, r, c)
    """

    n_masks = masks.shape[0]
    img_stack = np.dstack([img] * n_masks)
    img_stack = img_stack.swapaxes(2, 1).swapaxes(1, 0)
    assert img_stack.shape == masks.shape, "img_stack: {}, masks: {}".format(img_stack.shape, masks.shape)
    return img_stack * masks
