
import harfbin.util.image_utils as u
import cv2
import numpy as np
from harfbin.util.decorators import static_vars

def mean_convexity_defect_distance(contour, hull_point_indices): 
    if len(contour) > 3 and len(hull_point_indices) > 2: 
        defects = cv2.convexityDefects(contour, hull_point_indices)
        if defects is not None and defects.size > 0:
            mean_defect_dist = defects[:, :, 3].mean()
            return mean_defect_dist / 256.0
    # if any of the conditions are not met
    return 0


CONTOUR_FEATURE_COLUMNS = [("x", "int"),
                           ("y", "int"),
                           ("width", "int"),
                           ("height", "int"),
                           ("area", "int"),
                           ("relative_width", "float"),
                           ("relative_height", "float"),
                           ("relative_area", "float"),
                           ("hull_area", "float"),
                           ("relative_hull_area", "float"),
                           ("rect_area", "float"),
                           ("relative_rect_area", "float"),
                           ("aspect_ratio", "float"),
                           ("extent", "float"),
                           ("solidity", "float"),
                           ("convexity_defect_dist", "float"),
                           ("normalized_convexity_defect_distance", "float"),
                           ("major_axis_length", "float"),
                           ("minor_axis_length", "float"),
                           ("orientation_angle", "float"),
                           ("color_mean", "float"),
                           ("color_std", "float"),
                           ("distance_transform_mean", "float"),
                           ("distance_transform_std", "float"),
                           ("distance_transform_max", "float"),
                           ("num_holes", "float"),
                           ("hierarchy_level", "float"),
                           ("class", "int")]

def contour_features(contour, holes, level, image, row=None): 
    """
    Features this function returns: 
    x
    y
    width
    height
    area
    relative_width
    relative_height
    relative_area
    hull_area
    relative_hull_area
    rect_area
    relative_rect_area
    aspect_ratio
    extent
    solidity
    convexity_defect_dist
    normalized_convexity_defect_distance
    major_axis_length
    minor_axis_length
    orientation_angle
    color_mean
    color_std
    distance_transform_mean
    distance_transform_std
    distance_transform_max
    num_holes
    hierarchy_level
    """
    
    x, y, width, height = cv2.boundingRect(contour)
    # image is grayscale
    # LOG.debug("=== image.shape ===")
    # LOG.debug(image.shape)
    
    image_width, image_height = image.shape
    relative_width = width / image_width
    relative_height = height / image_height
    area = cv2.contourArea(contour)
    image_area = (image_width * image_height)
    relative_area = area / image_area
    hull_point_indices = cv2.convexHull(contour, returnPoints=False)
    hull = contour[np.transpose(hull_point_indices)]
    # LOG.debug("=== len(hull) ===")
    # LOG.debug(len(hull))
    # LOG.debug("=== hull ===")
    # LOG.debug(hull)
    
    hull_area = cv2.contourArea(hull[0])
    relative_hull_area = hull_area / image_area
    rect_area = width * height
    relative_rect_area = rect_area / image_area
    aspect_ratio = width / height
    extent = float("inf") if rect_area == 0 else area / rect_area
    solidity = float("inf") if hull_area == 0 else area / hull_area
    convexity_defect_dist = mean_convexity_defect_distance(contour, hull_point_indices)
    normalized_convexity_defect_distance = convexity_defect_dist / hull_area if hull_area != 0 else float("inf")
    if contour.size > 8: 
        # LOG.debug("=== contour.size ===")
        # LOG.debug(contour.size)
        
        (ellipse_x, ellipse_y), (major_axis_length, minor_axis_length), orientation_angle = cv2.fitEllipse(contour)
    else: 
        major_axis_length = float("inf")
        minor_axis_length = float("inf")
        orientation_angle = float("inf")
    # draw outer with white and holes with black for the mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    cv2.drawContours(mask, holes, -1, 0, -1)
    pixel_points = np.nonzero(mask)
    color_values = image[pixel_points]
    # LOG.debug("=== color_values ===")
    # LOG.debug(color_values)
    
    color_mean = color_values.mean()
    color_std = color_values.std()

    # # LOG.debug("=== mask.nonzero() ===")
    # # LOG.debug(mask.nonzero())
    # LOG.debug("=== pixel_points ===")
    # LOG.debug(pixel_points)
    
    min_x = pixel_points[0].min()
    max_x = pixel_points[0].max()
    min_y = pixel_points[1].min()
    max_y = pixel_points[1].max()
    
    distance_transform_roi = mask[min_x:(max_x+1), min_y:(max_y+1)]
    # LOG.debug("=== distance_transform_roi ===")
    # LOG.debug(distance_transform_roi)
    
    distance_transform = cv2.distanceTransform(distance_transform_roi, DISTANCE_TRANSFORM_METRIC, DISTANCE_TRANSFORM_BLOCKSIZE)
    distance_transform_values = distance_transform[distance_transform.nonzero()]
    distance_transform_mean = distance_transform_values.mean()
    distance_transform_std = distance_transform_values.std()
    distance_transform_max = distance_transform_values.max()
    
    num_holes = len(holes)
    hierarchy_level = level

    if row is not None: 
        row["x"] = x
        row["y"] = y
        row["width"] = width
        row["height"] = height
        row["area"] = area
        row["relative_width"] = relative_width
        row["relative_height"] = relative_height
        row["relative_area"] = relative_area
        row["hull_area"] = hull_area
        row["relative_hull_area"] = relative_hull_area
        row["rect_area"] = rect_area
        row["relative_rect_area"] = relative_rect_area
        row["aspect_ratio"] = aspect_ratio
        row["extent"] = extent
        row["solidity"] = solidity
        row["convexity_defect_dist"] = convexity_defect_dist
        row["normalized_convexity_defect_distance"] = normalized_convexity_defect_distance
        row["major_axis_length"] = major_axis_length
        row["minor_axis_length"] = minor_axis_length
        row["orientation_angle"] = orientation_angle
        row["color_mean"] = color_mean
        row["color_std"] = color_std
        row["distance_transform_mean"] = distance_transform_mean
        row["distance_transform_std"] = distance_transform_std
        row["distance_transform_max"] = distance_transform_max
        row["num_holes"] = num_holes
        row["hierarchy_level"] = hierarchy_level
    
    return (x, y, width, height, area, relative_width, relative_height,
            relative_area, hull_area, relative_hull_area, rect_area,
            relative_rect_area, aspect_ratio, extent, solidity,
            convexity_defect_dist, normalized_convexity_defect_distance,
            major_axis_length, minor_axis_length, orientation_angle, color_mean,
            color_std, distance_transform_mean, distance_transform_std,
            distance_transform_max, num_holes, hierarchy_level)


@static_vars(mser = cv2.MSER_create())
def mser_components(img): 
    regions = mser_components.mser.detectRegions(img, None)
    return regions


def word_regions(line, 
                 font="Amiri", 
                 dpi=600, 
                 background=(0, 0, 0), 
                 foreground=(255, 255, 255), 
                 **kwargs):
    text_img = u.get_word_image(line, font, dpi, background, foreground)
    regions = mser_components(text_img)
    return text_img, regions
    
def generate_word_regions(text_lines): 
    # input_f = open(input_text_file)
    # input_lines = input_f.readlines()
    img_dict = {}
    region_dict = {}
    for line in text_lines: 
        key = line.strip()
        img, regions = word_regions(key)
        img_dict[key] = img
        region_dict[key] = regions
    return img_dict, region_dict

