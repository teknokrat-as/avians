import numpy as np
import cv2
import math as m
import os
from datetime import datetime as dt
import argparse

import avians.util.image as aui
import avians.detect.train_rnn as tr
import avians.detect.generate_dataset as gd

import logging
LOG = logging.getLogger(__name__)

# Stroke width transform
def stroke_width_transform(edge, gradx, grady, dol, swt, ray, gry_img):
    row, col = gry_img.shape
    iteray = -1
    for i in range(row):
        for j in range(col):
            if edge[i, j] > 0:
                iteray += 1
                ray.append([[i, j]])
                cur_row = i
                cur_col = j
                cur_row_new = i
                cur_col_new = j
                g_x = gradx[i, j]
                g_y = grady[i, j]
                mag = m.sqrt(g_x*g_x + g_y*g_y)
                if dol:
                    g_x /= -mag
                    g_y /= -mag
                else:
                    g_x /= mag
                    g_y /= mag
                # print(G_x)
                # print(G_y)
                while True:
                    cur_row += g_y
                    cur_col += g_x
                    if (m.floor(cur_row) != cur_row_new) or (m.floor(cur_col) != cur_col_new):
                        cur_row_new = m.floor(cur_row)
                        cur_col_new = m.floor(cur_col)
                        # check if pixel is outside boundary of image
                        if cur_col_new < 0 or cur_col_new >= col or cur_row_new < 0 or cur_row_new >= row:
                            break
                        ray[iteray].append([cur_row_new, cur_col_new])
                        if edge[cur_row_new, cur_col_new] > 0:
                            g_xt = gradx[cur_row_new, cur_col_new]
                            g_yt = grady[cur_row_new, cur_col_new]
                            mag1 = m.sqrt(g_xt * g_xt + g_yt * g_yt)
                            if dol:
                                g_xt /= -mag1
                                g_yt /= -mag1
                            else:
                                g_xt /= mag1
                                g_yt /= mag1
                            if m.atan2(g_y, g_x) < 0:
                                teta1 = 2*np.pi + m.atan2(g_y, g_x)
                            else:
                                teta1 = m.atan2(g_y, g_x)

                            if m.atan2(g_yt, g_xt) < 0:
                                teta2 = 2*np.pi + m.atan2(g_yt, g_xt)
                            else:
                                teta2 = m.atan2(g_yt, g_xt)

                            if 2*np.pi/3 <= abs(teta1 - teta2) <= 4*np.pi/3:
                                length = m.sqrt((cur_row_new - ray[iteray][0][0])**2 + (cur_col_new - ray[iteray][0][1])**2)
                                for k in range(len(ray[iteray])):
                                    if swt[ray[iteray][k][0], ray[iteray][k][1]] < 0:
                                        swt[ray[iteray][k][0], ray[iteray][k][1]] = length
                                    else:
                                        swt[ray[iteray][k][0], ray[iteray][k][1]] = min(length, swt[ray[iteray][k][0], ray[iteray][k][1]])

                            break


# Change the stroke width values with integers which are floor of stroke width transform values
def swt_int(str_image, gry_img):
    row, col = gry_img.shape
    swt_org = np.zeros((row, col), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            if str_image[i, j] == -1:
                swt_org[i, j] = 0
            else:
                swt_org[i, j] = np.floor(str_image[i, j])
    return swt_org


# Function which signs the very big or very small or so thin contours..
def sign_high_contours(ctr, hrc):
    for i in range(len(ctr)):
        x, y, w, h = cv2.boundingRect(ctr[i])
        aspect_ratio = float(w)/h
        if (h <= 5 and w <= 5) or aspect_ratio < 0.1 or aspect_ratio > 10 or w > 300 or h > 300:
            hrc[0, i, 1] = -2


# This function signs contours which includes more than 3 contours..
def including_contours(hcy):
    vct = np.zeros(len(hcy[0]))
    for j in range(len(vct)):
        m = hcy[0, j, 3]
        vct[m] += 1
    for l in range(len(vct)):
        if vct[l] >= 3:
            k = l
            while k != -1:
                hcy[0, k, 1] = -2
                k = hcy[0, k, 3]


# This function signs contours whose stroke width variance are too big..
def improper_variance(swt_int, cont, hrcy, gry_img):
    for i in range(len(cont)):
        if hrcy[0, i, 1] != -2:
            mask = np.zeros(gry_img.shape, np.uint8)
            cnt = cont[i]
            mask = cv2.drawContours(mask, [cnt], 0, 255, -1)
            mask1 = cv2.bitwise_and(swt_int, swt_int, mask=mask)
            mean_val1 = cv2.mean(swt_int, mask=mask1)
            mask2 = mask1*((mask1-mean_val1[0])**2)/(mask1+0.0000001)
            variance = cv2.mean(mask2, mask = mask1)
            if variance[0] >= 200:
                hrcy[0, i, 1] = -2


# This function signs contours whose ratio of equivalent diameter and median stroke width is greater than 15..
def diameter_median(ct, hc, int_str, gry_img):
    median = []
    for i in range(len(ct)):
        if hc[0, i, 1] != -2:
            cnt = ct[i]
            area = cv2.contourArea(cnt)
            equiv_diameter = np.sqrt(4*area/np.pi)
            mask = np.zeros(gry_img.shape, np.uint8)
            mask = cv2.drawContours(mask,  [cnt],  0,  255,  -1)
            mask1 = cv2.bitwise_and(int_str, int_str, mask=mask)
            pixel_points = np.transpose(np.nonzero(mask1))
            row1, col1 = pixel_points.shape
            for j in range(row1):
                median.append(int_str[pixel_points[j, 0], pixel_points[j, 1]])
            median.sort()
            ln = len(median)
            if ln != 0:
                if ln % 2 == 1:
                    av = int((ln-1)/2)
                    mn = median[av]
                else:
                    av1 = int(ln/2)
                    av2 = int((ln-2)/2)
                    mn = (median[av1] + median[av2])/2
                if mn != 0:
                    if (equiv_diameter/mn) >= 15:
                        hc[0, i, 1] = -2


# This function deletes contours which we signed until now.
def deleting_contours(cs, hy):
    m = 0
    while m < len(cs):
        if hy[0, m, 1] == -2:
            hy = np.delete(hy, m, 1)
            cs.pop(m)
            m -= 1
        m += 1


# This function finds boundary points of contours..
def boundary_points(cntrs):
    mst_vls = np.zeros((len(cntrs), 8))
    for i in range(len(cntrs)):
        cnt = cntrs[i]
        leftmost1 = tuple(cnt[cnt[:, :, 0].argmin()][0])
        mst_vls[i, 0] = leftmost1[0]
        mst_vls[i, 1] = leftmost1[1]
        rightmost1 = tuple(cnt[cnt[:, :, 0].argmax()][0])
        mst_vls[i, 2] = rightmost1[0]
        mst_vls[i, 3] = rightmost1[1]
        topmost1 = tuple(cnt[cnt[:, :, 1].argmin()][0])
        mst_vls[i, 4] = topmost1[0]
        mst_vls[i, 5] = topmost1[1]
        bottommost1 = tuple(cnt[cnt[:, :, 1].argmax()][0])
        mst_vls[i, 6] = bottommost1[0]
        mst_vls[i, 7] = bottommost1[1]
    return mst_vls


# This function finds two contours which can be a part of word..
def two_contours(countr, bound_pts):
    chain = []
    for i in range(len(countr)):
        bot = bound_pts[i, 7]
        top = bound_pts[i, 5]
        left = bound_pts[i, 0]
        right = bound_pts[i, 2]
        w = right - left
        h = bot - top
        for j in range(len(countr)):
            if bound_pts[j, 5] < bot and bound_pts[j, 7] > top and bound_pts[j, 0] > left:
                h1 = bound_pts[j, 7] - bound_pts[j, 5]
                d1 = bound_pts[j, 0] - left
                if h/2 <= h1 <= 2*h and d1 <= 3*w:
                    chain.append([i, j])
    return chain


# According to our two contour chains,  some contour has more than one contour after its,  so we should combine them..
def combine_contours(chn):
    i = 0
    while i < len(chn) - 1:
        if chn[i][0] == chn[i+1][0]:
            chn[i].append(chn[i+1][1])
            chn.pop(i+1)
            i -= 1
        i += 1


# This function create chains which can be word..
def make_chains(chan):
    for i in range(len(chan)):
        j = 1
        while j < len(chan[i]):
            for k in range(len(chan)):
                if chan[i][j] == chan[k][0]:
                    chan[i].extend(chan[k])
                    chan[k] = [-1]
            j += 1


# This function delete chains which is equal to "-1" or has less contour than 3..
def clear_chain(chain, chain_length):
    i = 0
    while i < len(chain):
        chain[i] = list(set(chain[i]))
        if chain[i] == [-1] or len(chain[i]) < chain_length:
            chain.pop(i)
            i -= 1
        i += 1


# This function finds chains's boundaries..
def word_boundaries(chain, most_val, gry_img):
    row, col = gry_img.shape
    length = len(chain)
    right = np.zeros(length)
    down = np.zeros(length)
    left = col*np.ones(length)
    top = row*np.ones(length)
    for i in range(length):
        for j in range(len(chain[i])):
            if most_val[chain[i][j], 0] < left[i]:
                left[i] = most_val[chain[i][j], 0]
            if most_val[chain[i][j], 2] > right[i]:
                right[i] = most_val[chain[i][j], 2]
            if most_val[chain[i][j], 5] < top[i]:
                top[i] = most_val[chain[i][j], 5]
            if most_val[chain[i][j], 7] > down[i]:
                down[i] = most_val[chain[i][j], 7]
    return left, right, top, down


# This function eliminates rectangles which is inside the other rectangles..
def including_rectangle_deleting(left1, right1, top1, down1, left2, right2, top2, down2):
    left_org = np.concatenate((left1, left2))
    right_org = np.concatenate((right1, right2))
    top_org = np.concatenate((top1, top2))
    down_org = np.concatenate((down1, down2))
    i = 0
    while i < len(left_org):
        for j in range(len(left_org)):
            if i != j and down_org[i] <= down_org[j] + 5 and top_org[i] + 5 >= top_org[j]:
                if left_org[j] <= left_org[i] <= right_org[j] or left_org[j] <= right_org[i] <= right_org[j]:
                    left_org[j] = min(left_org[j], left_org[i])
                    right_org[j] = max(right_org[j], right_org[i])
                    top_org[j] = min(top_org[j], top_org[i])
                    down_org[j] = max(down_org[j], down_org[i])
                    left_org = np.delete(left_org, i, 0)
                    right_org = np.delete(right_org, i, 0)
                    top_org = np.delete(top_org, i, 0)
                    down_org = np.delete(down_org, i, 0)
                    i -= 1
                    break
        i += 1
    return left_org, right_org, top_org, down_org

# Here we draw the boundaries of words..
def boundary_drawing(image, lft, rght, tp, dwn):
    for i in range(len(lft)):
        image = cv2.rectangle(image, (int(lft[i]), int(tp[i])), (int(rght[i]), int(dwn[i])), (0, 0, 255), 1)
    return image


def detect_text(img_filename, dark_on_light=1, chain_length=3, output_image=False, output_mask=False):
    """

    Detects text on an image and returns a set of rectangle coordinates

    :param img_filename: The image to be processed

    :param dark_on_light: 1 (Dark text on light BG)
                          0 (Light text on dark BG)
                          2 (Run both and combine)


    :param chain_length: Number of chains to constitute a text

    """

    if dark_on_light == 2: 
        detect_text_inner(img_filename, 0, chain_length, output_image, output_mask)
        detect_text_inner(img_filename, 1, chain_length, output_image, output_mask)
    else: 
        detect_text_inner(img_filename, dark_on_light, chain_length, output_image, output_mask)

    #detect_text_via_components(img_filename, chain_length)

def draw_possibly_textual_components(img, contour_tree, up_to = 5): 
    def all_drawables(contour_tree): 
        drawables = []
        LOG.debug("=== len(contour_tree) ===")
        LOG.debug(len(contour_tree))
        for contour, children, level in contour_tree: 
            if level < up_to:
                # LOG.debug("=== level ===")
                # LOG.debug(level)
                # LOG.debug("=== len(contour) ===")
                # LOG.debug(len(contour))
                drawables.append((contour, level))
                # then draw the children
            for child in children: 
                drawables += all_drawables(child)
        return drawables
    drawables = all_drawables(contour_tree)
    # LOG.debug("=== drawables ===")
    # LOG.debug(drawables)
    for level in range(up_to, 0, -1):
        # LOG.debug("=== level ===")
        # LOG.debug(level)
        cv2.drawContours(img, [d[0] for d in drawables if d[1] == level], 0, level * 25, -1)
    return img

def get_contour_image(gray_img):
    img = aui.copy_with_frame(gray_img, 2)
    img = aui.adaptive_mean(img)
    contour_res = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = contour_res[1]
    LOG.debug("=== len(contours) ===")
    LOG.debug(len(contours))
    hierarchy = contour_res[2][0]
    contour_tree = aui.get_contour_tree(contours, 
                                        hierarchy, 
                                        1)
    result_img = np.ones_like(img) * 255
    draw_possibly_textual_components(result_img, contour_tree)
    return result_img
        
def detect_text_via_components(img_filename, chain_length): 
    img = cv2.imread(os.path.expandvars(img_filename), 0)
    row, col = img.shape
    secs = dt.now().strftime("%s")
    gaussian_image = cv2.GaussianBlur(img, (3, 3), 0)
    contour_image = get_contour_image(gaussian_image)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "contour_image", secs), contour_image)

def detect_text_inner(img_filename, dark_on_light, chain_length, output_image, output_mask):
    LOG.debug("=== img_filename ===")
    LOG.debug(img_filename)
    
    img = cv2.imread(os.path.expandvars(img_filename))
    row, col, chan = img.shape
    secs = dt.now().strftime("%s")
    LOG.debug("=== (row, col, chan) ===")
    LOG.debug((row, col, chan))
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "gray_image", secs), gray_image)
    # Edge detection
    edge_image = cv2.Canny(gray_image, 100, 200, True)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "edge_image", secs), edge_image)
    # Create gradient X, gradient Y
    gaussian_image = gray_image/255
    gaussian_image = cv2.GaussianBlur(gaussian_image, (5, 5), 0)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "gaussian_image", secs), gaussian_image)
    gradientx = cv2.Sobel(gaussian_image, cv2.CV_64F, 1, 0, ksize=-1)
    gradienty = cv2.Sobel(gaussian_image, cv2.CV_64F, 0, 1, ksize=-1)
    gradientx = cv2.GaussianBlur(gradientx, (3, 3), 0)
    gradienty = cv2.GaussianBlur(gradienty, (3, 3), 0)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "gradientx", secs), gradientx)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "gradienty", secs), gradienty)
# Create SWTImage and Ray vectors
    swt_image = -1*np.ones((row, col))
    ray = []

    # Finding swtimage and Ray vectors and swtimageorg(floor of swtimage)
    stroke_width_transform(edge_image, gradientx, gradienty, dark_on_light, swt_image, ray, gray_image)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "swt_image", secs), swt_image)

    # After Stroke width transform Swtimage has values which are not integer.
    # So i changed the values with floor of numbers and "0" for "-1" values...
    swt_image_int = swt_int(swt_image, gray_image)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "swt_image_int", secs), swt_image_int)
    
    # Finding contours of image(After working is finished with swtimage(stroke width image),
    # I found the connected components of image(contours)
    # thres_param = cv2.THRESH_BINARY_INV if dark_on_light else cv2.THRESH_BINARY
    gray_image = aui.adaptive_mean(gray_image) 
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "gray_image", secs), gray_image)
    
    # gray_image_copy = np.copy(gray_image)
    contour_image, contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "contour_image", secs), contour_image)
    
    # Finding contours which have high width, height and ratio of width over high.
    # After finding contours,  I started to find contours which can't be text..
    # sign_high_contours(contours, hierarchy)
    
    # Finding contours which have 3 or greater than 3 child, I continued the working of finding non-text contours..
    # In here I found contours which includes 3 or more than 3 contours..
    # including_contours(hierarchy)
    
    # Finding contours whose stroke width variance is too big or too small(In here our stroke width transform provide us
    # property for eliminating non-text contours) After stroke width transform we use stroke width transform first here..
    improper_variance(swt_image_int, contours, hierarchy, gray_image)

    # Finding contours whose ratio of equivalent diameter and stroke width median value is greater than 10...
    # And here is the other place where we use stroke width. I firstly find diameter of the contours and compare its with
    # Its median stroke width(It means contour's stroke width's median value
    diameter_median(contours, hierarchy, swt_image_int, gray_image)

    # Deleting contours which is not text; We think until now.. (And in here,  I delete the contours which are non-text
    # according to the above test(I signed contours while i test them and now i deleted them..)
    deleting_contours(contours, hierarchy)

    # Finding most values of contours(I found the boundary points of contours(top boundary,  left boundary, down and right))
    most_values = boundary_points(contours)

    # Finding chains; firstly, i found the only two components which can be word..
    chain = two_contours(contours, most_values)

    # According to our two contour chains, some contour has more than one contour after its, so we should combine them..
    combine_contours(chain)

    # After all we make chains which can be word..
    make_chains(chain)

    # In here we eliminate chains which can't be word..
    clear_chain(chain, chain_length)

    # After creating chains(words),  we finds boundary points of chains(words)..
    left, right, top, down = word_boundaries(chain, most_values, gray_image)
    # left, right, top, down = including_rectangle_deleting(left_dark, right_dark, top_dark, down_dark, left_white, right_white, top_white, down_white)

    # Now drawing words boundaries on the image..
    result_img = boundary_drawing(img, left, right, top, down)
    cv2.imwrite("{}-{}-{}.png".format(img_filename, "result_img.png", secs), result_img)
    path = img_filename.split('/')
    path.append("_" + path.pop())
    img_path = "/".join(path)
    cv2.imwrite(img_path, img)
    return img_path

#detect_text("/tmp/_hyr.jpg")

def detect_text_with_cnn(model_name, imgf, rows=32, cols=32): 
    img = cv2.imread(imgf)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    model = tr.load_cnn_model(model_name)
    large, contours = aui.enlarge_and_get_contours(gray_img)
    zoom_ratio = large.shape[0] // img.shape[0]

    features = np.zeros(shape=(len(contours), 1, rows, cols), dtype=np.uint8)
    locations = np.zeros(shape=(len(contours), 4), dtype=np.int)

    for i, contour in enumerate(contours): 
        features[i, 0] = gd.resized_contour_region(large, contour, rows, cols)
        cnt, holes, level = contour
        locations[i] = cv2.boundingRect(cnt)

    predictions = model.predict_proba(features, batch_size=128, verbose=1)
    indices = (predictions > 0.01).nonzero()[0]
    dezoomed = locations / zoom_ratio
    
    for loc in dezoomed[indices]: 
        r1 = (int(loc[0]), int(loc[1]))
        r2 = (int(loc[0]) + int(loc[2]), int(loc[1]) + int(loc[3]))
        cv2.rectangle(img, r1, r2, (0, 0, 255), 1)        

    return img
    # for ll in large_locations:
    #     l = ll // zoom_ratio
    #     print(ll)
    #     print(l)
    #     cv2.rectangle(img, l[0], l[1], l[2], l[3], (0, 0, 255), 1)        
    # return img
    
    

def main(): 
    parser = argparse.ArgumentParser(description="""
    Detects textual regions on the given image using the supplied model generated by train_rnn
    """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('model', help="Keras model name to detect text (without .json or .h5 extension)")
    parser.add_argument('image', help="Image that we are going to detect the text")
    parser.add_argument('--output', help="File to save the image", default="text-detection-result-{}.png".format(dt.now().strftime("%F-%T")))

    args = vars(parser.parse_args())
    print(args)
    result = detect_text_with_cnn(args['model'], args['image'])
    cv2.imwrite(args['output'], result)
    print("Result Saved To: {}".format(args['output']))

if __name__ == '__main__': 
    main()


"""
print(chain_white)
contours_new = []
for i in range(len(chain)) :
    for j in range(len(chain[i])) :
        contours_new.append(contours[chain[i][j]])
#np.set_printoptions(None,1068354)
#print(chain)
#mask5 = np.zeros(gray_image1.shape,np.uint8)
#mask5 = cv2.drawContours(mask5,contours,-1,255,-1)
#mask6 = np.zeros(gray_image1.shape,np.uint8)
#mask6 = cv2.drawContours(mask6,contours1,-1,255,-1)
#cv2.imshow('mask6',mask6)
"""
