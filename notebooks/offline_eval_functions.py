import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.morphology import skeletonize
from pycocotools.coco import COCO

import numpy as np
import cv2
import matplotlib.pyplot as plt

def connect_clean_mask(mask, buffer_radius=10):
    """
    Connect the tails to the main body and clean out noise (outside of 10px buffer).

    Parameters:
    - mask (np.array): Input binary mask (2D NumPy array).
    - buffer_radius (int): The radius of the buffer around the main mask to include secondary masks.

    Returns:
    - connected_mask (np.array): The mask with secondary masks connected to the main mask.
    """
    # identify main mask (shark)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    main_mask = np.zeros_like(mask)
    cv2.drawContours(main_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # create buffer on shark body
    buffer_mask = cv2.dilate(main_mask, np.ones((buffer_radius, buffer_radius), np.uint8))

    # identify secondary masks that intersect buffer zone (tails)
    secondary_masks = []
    for contour in contours:
        if contour is not largest_contour:  # Skip the largest contour (main mask)
            sec_mask = np.zeros_like(mask)
            cv2.drawContours(sec_mask, [contour], -1, 255, thickness=cv2.FILLED)
            intersection = cv2.bitwise_and(sec_mask, buffer_mask) # check for intersection
            if np.any(intersection == 255):  # if intersection, preserve
                secondary_masks.append(sec_mask)

    # find closest points in body and tail masks
    def connect_masks_by_pixel(main_mask, sec_mask):
        main_contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sec_contours, _ = cv2.findContours(sec_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        main_contour = max(main_contours, key=cv2.contourArea)
        sec_contour = max(sec_contours, key=cv2.contourArea)

        min_distance = float('inf')
        point_main, point_sec = None, None
        for pt1 in main_contour:
            for pt2 in sec_contour:
                distance = np.linalg.norm(pt1 - pt2)
                if distance < min_distance:
                    min_distance, point_main, point_sec = distance, pt1[0], pt2[0]

        return point_main, point_sec

    # draw one-pixel line between points
    def draw_connection_line(image, point1, point2):
        cv2.line(image, tuple(point1), tuple(point2), (255), thickness=1)

    # combine masks with connection line
    connected_mask = np.copy(main_mask) 
    for sec_mask in secondary_masks:
        connected_mask = cv2.bitwise_or(connected_mask, sec_mask)

    # draw connection line 
    for sec_mask in secondary_masks:
        point_main, point_sec = connect_masks_by_pixel(main_mask, sec_mask)
        if point_main is not None and point_sec is not None:
            draw_connection_line(connected_mask, point_main, point_sec)

    return connected_mask

def skeleton(mask):
    """input mask, get skeleton"""
    mask = np.array(mask)
    skeleton = skeletonize(mask)
    return skeleton
    
def skeleton_length(mask): ### you're going to have to work on this here
    """input mask, get skeleton length in pixels"""
    skeleton_length = np.sum(skeletonize(mask))
    return skeleton_length

def mask_area(mask):
    """input mask, get area (pixels)"""
    area = np.sum(mask)
    return area

def get_cross_sectional_lengths(mask):
    """input mask, get cross sectional lengths down the body,
    in 1% increments, starting at pixel 20/ending at pixel -20"""
    medial_line = np.column_stack(np.where(skeleton(mask))) # extract medial line

    directions = []
    for i in range(1, len(medial_line) - 1):
        # Get the neighboring points on the medial line to calculate direction
        p1 = medial_line[i - 1]
        p2 = medial_line[i + 1]
        
        # Direction vector (perpendicular to the line connecting p1 and p2)
        direction = np.array([p2[1] - p1[1], p1[0] - p2[0]])  # 90-degree rotation (counter-clockwise)
        direction = direction / np.linalg.norm(direction)  # Normalize direction
        directions.append(direction)
    
    cross_sections = []  # This will store the final section lengths, one for each medial point
    for i, point in enumerate(medial_line[1:-1]):
        direction = directions[i]
        
        section_length = 0
        for offset in np.linspace(-20, 20, num=100): 
            x_offset = int(round(point[0] + offset * direction[0]))  # Ensure integer pixel coordinates
            y_offset = int(round(point[1] + offset * direction[1]))  # Ensure integer pixel coordinates

            # Check if the offset point is inside the mask
            if 0 <= x_offset < mask.shape[0] and 0 <= y_offset < mask.shape[1]:  # Ensure inside bounds
                if mask[x_offset, y_offset]:  # Check if it's a body pixel
                    section_length += 1  # Count it as part of the cross-section

        cross_sections.append(section_length)  # Store the final length for this point

    return cross_sections, directions

def find_intersection(mask, start_point, direction, max_distance=50):
    """ Finds intersection point with the mask along a direction from a start point """
    
    if np.isnan(start_point[0]) or np.isnan(start_point[1]):
        return None  # Skip this calculation if the start point is NaN

    if np.isnan(direction[0]) or np.isnan(direction[1]):
        return None  # Skip this calculation if the direction is NaN

    if np.linalg.norm(direction) == 0:
        return None 
    
    for dist in np.linspace(1, max_distance, num=max_distance):
        x_offset = int(round(start_point[0] + dist * direction[0]))
        y_offset = int(round(start_point[1] + dist * direction[1]))
        if 0 <= x_offset < mask.shape[0] and 0 <= y_offset < mask.shape[1]:
            if mask[x_offset, y_offset] == 0:  # Transition from 1 to 0, intersection found
                return (x_offset, y_offset)
    
    return None


def get_cross_sectional_points(mask, smooth_window=20):
    """get the two cross sectional points that intersect the mask 
    for easy computation and storing. 
    included functionality for different sized masks - proportional smoothing window""" 
    medial_line = np.column_stack(np.where(skeleton(mask)))
    total_length = len(medial_line)
    cross_section_points = []
    smooth_window = round((total_length)*(smooth_window/100)) # proportional smoothing window

    for i in range(0, total_length, max(1, total_length // 100)): # 1% interval traversal
        point = medial_line[i]
        
        if i > smooth_window and i < total_length - smooth_window:
            p1 = medial_line[i - smooth_window]
            p2 = medial_line[i + smooth_window]
            direction = np.array([p2[1] - p1[1], p1[0] - p2[0]])  # Perpendicular direction
            direction = direction / np.linalg.norm(direction)  
        
            forward_intersection = find_intersection(mask, point, direction)
            backward_intersection = find_intersection(mask, point, -direction)
        
            if forward_intersection and backward_intersection:
                cross_section_points.append((forward_intersection, backward_intersection))

    return cross_section_points

def get_widest_cross_section(cross_section_points):
    """Get the three widest cross sections (of 1%s) and their average distance."""
    distances = []
    
    for forward, backward in cross_section_points:
        distance = np.linalg.norm(np.array(forward) - np.array(backward))
        distances.append((distance, (forward, backward)))

    distances.sort(reverse=True, key=lambda x: x[0])
    top_three = distances[:3] # top three widths
    top_three_pairs = [pair for _, pair in top_three]
    top_three_distances = [distance for distance, _ in top_three]
    average_distance = np.mean(top_three_distances)
    
    return top_three_pairs, average_distance


def get_mask_dims(annotations_path):
    """input annotations, get the bounding box
    dimensions of the annotation"""
    coco =  COCO(annotations_path)
    annotation_ids = coco.getAnnIds()
    annotations = coco.loadAnns(annotation_ids)

    mask_data = []
    for ann in annotations:
        image_id = ann['image_id']
        mask = coco.annToMask(ann)
        img_height, img_width = mask.shape # extract img dims
        bbox_width, bbox_height = ann['bbox'][2], ann['bbox'][3] # extract bbox dims

        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']

        mask_data.append({'FileName':file_name, 'img_height':img_height, 'img_width':img_width,
            'mask_height': bbox_height, 'mask_width': bbox_width})
    
    df = pd.DataFrame(mask_data)
    return df

###
def process_biometrics2(mask_list): 
    """takes in existing masks (annotations) and performs computations"""
    data = []  # store rows for df
    for mask in mask_list:
        mask = np.array(mask)
        skeleton_TL = skeleton_length(mask) # extract medial tl
        body_area = mask_area(mask) # extract body area
        cross_sectional_points = get_cross_sectional_points(mask) # extract cx points
        body_span = get_widest_cross_section(cross_sectional_points)[1] # extract max span
        image_name = file.replace('pred_', '').replace('.png', '.JPG') # revert image name
        data.append((image_name, skeleton_TL, body_area, body_span)) # append tuple 

    df = pd.DataFrame(data, columns=['filename', 'skeleton_TL', 'body_area', 'body_span']) # construct df
        
    return(df)


def process_biometrics(root_predictions, pred_files): 
    """add morphometric variables"""
    data = []  # store rows for df
    for file in pred_files:
        mask_path = os.path.join(root_predictions, file) # full path 
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = connect_clean_mask(mask) # connect tails, clean out artifacts
        skeleton_TL = skeleton_length(mask) # extract medial tl
        body_area = mask_area(mask) # extract body area
        cross_sectional_points = get_cross_sectional_points(mask) # extract cx points
        body_span = get_widest_cross_section(cross_sectional_points)[1] # extract max span
        image_name = file.replace('pred_', '').replace('.png', '.JPG') # revert image name
        data.append((image_name, skeleton_TL, body_area, body_span)) # append tuple 

    df = pd.DataFrame(data, columns=['filename', 'skeleton_TL', 'body_area', 'body_span']) # construct df
        
    return(df)

def reconstruct_pixels_from_crop(df, crop_size, img_size, use_custom_crop):
    pixel_transf_factors = []
    for _, row in df.iterrows():
        relative_altitude = row['RelativeAltitude_x']  # change as needed
        img_width = row['ImageWidth_x']  # change as needed
        img_height = row['ImageHeight_x'] # change as needed

        if use_custom_crop:
            crop_size = compute_custom_crop_size(relative_altitude, img_width) # compute custom
        else:
            crop_size = crop_size # pull input

        if crop_size == 0: #if no crop - need to pull original image size
            scale_width = img_width/img_size
            scale_height = img_height/img_size
            pixel_transf_factor = (scale_width+scale_height)/2 # applying average h/w transf - this is not comprehensive!

        else: 
            img_size = img_size # pull input
            pixel_transf_factor = crop_size / img_size
            
        
        pixel_transf_factors.append(pixel_transf_factor)

    df['pixel_transf_factor'] = pixel_transf_factors

    return df

def compute_custom_crop_size(relative_altitude, img_width):
    """returns the custom crop size associated with the image
    ***you must change this here if you change it in the dataloader***
    """
    crop_size = 0 # initialize
    if 0 <= relative_altitude <= 30: # Low altitudes
        if img_width <= 3000: crop_size = 672
        elif 3000 < img_width <= 4000: crop_size = 672
        else: crop_size = 896 # img_width > 4000
    
    elif 30 < relative_altitude <= 50: # Medium altitudes
        if img_width <= 3000: crop_size = 448
        elif 3000 < img_width <= 4000: crop_size = 448
        else: crop_size = 672 # img_width > 4000
    
    elif 50 < relative_altitude <= 100: # High altitudes
        if img_width <= 3000: crop_size = 448
        elif 3000 < img_width <= 4000: crop_size = 448
        else: crop_size = 672 # img_width > 4000
        
    return crop_size

def photogrammetric_conversion(df):
    """converts pixels to cm using photogrammetry"""
    gsd =  df['GSD_cm']
    flight_transf = df['Flight_Transformation'] # flight transf
    df['TL_pixels_skeleton_transf'] = df['skeleton_TL']*df['pixel_transf_factor']
    df['body_span_transf'] = df['body_span']*df['pixel_transf_factor']

    skel_pixt = df['TL_pixels_skeleton_transf'] # medial line (pix)
    body_span = df['body_span_transf']

    df['Calibrated_Skel_Length_cm'] = (skel_pixt*gsd)/flight_transf # transf w/flight-based transf
    df['Calibrated_body_span_cm'] = (body_span*gsd)/flight_transf 
 
    return df

## plotting functions ##

def plot_cross_sections(mask, cross_section_points):
    """plot the cross sections across the body"""
    plt.imshow(mask, cmap='gray')
    
    for forward, backward in cross_section_points:
        plt.plot([forward[1], backward[1]], [forward[0], backward[0]], 'r-', alpha=0.5)  # Cross section line
    
    plt.title("Cross-Sectional Lines at 1% Intervals")
    plt.show()
    
def plot_widest_cross_section(mask, cross_section_points):
    """plot only the widest cross secton along the body"""
    # Find the widest cross section
    widest_pair, max_distance = get_widest_cross_section(cross_section_points)

    # Extract forward and backward points
    forward, backward = widest_pair[1]

    # Plot the mask
    plt.imshow(mask, cmap='gray')
    plt.title('Widest Cross Section on Mask')
    plt.axis('off')

    # Overlay the line connecting the forward and backward points
    plt.plot([forward[1], backward[1]], [forward[0], backward[0]], color='red', linewidth=2, linestyle='--')

    # Mark the points
    plt.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='blue', zorder=5)
    plt.show()

def plot_three_widest_cross_sections(mask, cross_section_points):####
    """plot top three widest cross sectons along the body"""
    # Find the top three widest cross section
    widest_pairs, max_distances = get_widest_cross_section(cross_section_points)

    # Plot the mask
    plt.imshow(mask, cmap='gray')
    plt.title('Widest Cross Section on Mask')
    plt.axis('off')

    # Extract forward and backward points
    for pair in widest_pairs:
        forward, backward = pair[0], pair[1]
        plt.plot([forward[1], backward[1]], [forward[0], backward[0]], color='red', linewidth=2, linestyle='--')
        plt.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='blue', zorder=5)

    plt.show()

def plot_mask_with_skeleton_and_cross_sections(img, mask):
    """plot the mask with both the skeleton and the cross sections overlaid"""
    mask = connect_clean_mask(mask) # connect tails, clean out artifacts
    
    #skeleton
    skel_plot = skeleton(mask)
    skeleton_coords = np.column_stack(np.where(skel_plot == 1))  # Get (y, x) coordinates of skeleton

    #widths
    cross_section_points = get_cross_sectional_points(mask, smooth_window=20)
    widest_pairs, max_distances = get_widest_cross_section(cross_section_points)

    # plotting two panels side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    fig.patch.set_facecolor('black')  # Set the entire figure background to black

    # plot mask and morphometrics
    ax1 = axes[0]
    ax1.set_facecolor('black')  # Set the background of the axes to black
    ax1.imshow(mask, cmap='gray', vmin=0, vmax=255, alpha=0.5)

    for y, x in skeleton_coords:
        ax1.plot(x, y, 'ro', markersize=1, alpha=0.8)  # Plot each skeleton point as a red dot

    for pair in widest_pairs:
        forward, backward = pair[0], pair[1]
        ax1.plot([forward[1], backward[1]], [forward[0], backward[0]], color='red', linewidth=2, linestyle='--')
        ax1.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='blue', zorder=5)

    ax1.set_title('Morphometrics Mask', color='white')  # Title in white for visibility
    ax1.axis('off')  # Hide axis

    # plot original image
    ax2 = axes[1]
    ax2.set_facecolor('black') 
    ax2.imshow(img, cmap='gray', vmin=0, vmax=255)
    
    ax2.set_title('Original Image', color='white')  # title for the original image
    ax2.axis('off')  # hide axis
    
    plt.tight_layout()
    return fig # return the plot so you can use it