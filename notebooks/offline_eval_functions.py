import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.morphology import skeletonize
from skimage import measure
from skimage.draw import polygon as sk_polygon
from pycocotools.coco import COCO
from skan import summarize, Skeleton
from skan.csr import skeleton_to_csgraph
from collections import deque
import cv2

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

    # handle empty masks
    if not np.any(mask):
        return np.zeros_like(mask)

    # identify main contour (shark)
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

def compute_extended_path(skeleton, mask, num_points_src=5, num_points_dst=5):
    """
    Computes the extended path of a skeleton (line) to the mask edges from both the source and destination coordinates.
    Args:
    - skeleton: The skeletonized mask (binary).
    - mask: The original binary mask.
    - num_points_src: Number of points leading up to the source coordinates to calculate direction.
    - num_points_dst: Number of points leading up to the destination coordinates to calculate direction.
    Returns:
    - src_coords: Source coordinates of the longest branch.
    - dst_coords: Destination coordinates of the longest branch.
    - shortest_path: The shortest path between the source and destination coordinates.
    - extended_path: List of extended points that forms the entire path.
    """
    def calculate_direction(points, reverse=False):
        if reverse:
            points = points[::-1]  # Reverse the points for src_coords to point towards
        directions = np.diff(points, axis=0)  # Difference between consecutive points
        avg_direction = np.mean(directions, axis=0)
        norm = np.linalg.norm(avg_direction)
        if norm == 0:  # Avoid division by zero if points are identical
            return np.array([0, 0])
        return avg_direction / norm

    def extend_line(point, direction, mask):
        extended_points = []
        current_point = np.array(point, dtype=np.float64)
        direction = np.array(direction, dtype=np.float64)
        while True:
            current_point += direction
            current_point_int = np.round(current_point).astype(int)
            if current_point_int[0] < 0 or current_point_int[1] < 0 or current_point_int[0] >= mask.shape[0] or current_point_int[1] >= mask.shape[1]:
                break
            extended_points.append(tuple(current_point_int))
            if mask[current_point_int[0], current_point_int[1]] == 0:
                break
        return extended_points

    def get_neighbors(point, coordinates):
        neighbors = []
        for coord in coordinates:
            if np.abs(coord[0] - point[0]) <= 1 and np.abs(coord[1] - point[1]) <= 1 and not np.array_equal(coord, point):
                neighbors.append(coord)
        return neighbors

    def bfs_shortest_path(src, dst, coordinates):
        queue = deque([[src]])
        visited = set([tuple(src)])
        while queue:
            path = queue.popleft()
            current_point = path[-1]
            if np.array_equal(current_point, dst):
                return path
            for neighbor in get_neighbors(current_point, coordinates):
                if tuple(neighbor) not in visited:
                    visited.add(tuple(neighbor))
                    queue.append(path + [neighbor])
        return None

    coordinates = np.argwhere(skeleton)
    
    # summarize the skeleton data and get the longest branch
    branch_data = summarize(Skeleton(skeleton), separator="_")
    longest_branch_idx = branch_data['euclidean_distance'].idxmax()
    longest_branch_data = branch_data.iloc[longest_branch_idx]
    
    src_coords = longest_branch_data[['image_coord_src_0', 'image_coord_src_1']].values.flatten()
    dst_coords = longest_branch_data[['image_coord_dst_0', 'image_coord_dst_1']].values.flatten()
    
    shortest_path = bfs_shortest_path(src_coords, dst_coords, coordinates)
    
    # Get direction towards src_coords and dst_coords
    src_points = np.array(shortest_path[:num_points_src])
    direction_from_src = calculate_direction(src_points, reverse=True)
    dst_points = np.array(shortest_path[-num_points_dst:])
    direction_from_dst = calculate_direction(dst_points, reverse=False)
    
    # Extend the path
    extended_src_points = extend_line(src_coords, direction_from_src, mask)
    extended_dst_points = extend_line(dst_coords, direction_from_dst, mask)
    
    # Combine extended points with the original shortest path
    extended_path = []
    if extended_src_points:
        extended_path.extend(extended_src_points[::-1])
    extended_path.extend(shortest_path)
    if extended_dst_points:
        extended_path.extend(extended_dst_points)

    return np.array(src_coords), np.array(dst_coords), np.array(shortest_path), np.array(extended_path)

def resample_line(extended_path, num_points=20):
    """
    Resample the medial line to evenly spaced points. 
    Args: extended path: returned by compute_extended_path().
    Returns: array of resampled points with num_points as defined above.
    """
    dist_cumulative = [0]  # Starting point has distance 0
    for i in range(1, len(extended_path)):
        dist = np.linalg.norm(extended_path[i] - extended_path[i - 1])
        dist_cumulative.append(dist_cumulative[-1] + dist)
    
    total_length = dist_cumulative[-1]  # Total length of the path
    target_distances = np.linspace(0, total_length, num_points)  # Evenly spaced distances
    resampled_points = []
    for target_distance in target_distances:
        for i in range(1, len(dist_cumulative)):
            if dist_cumulative[i] >= target_distance:
                p1 = extended_path[i - 1]
                p2 = extended_path[i]
                segment_length = dist_cumulative[i] - dist_cumulative[i - 1]
                if segment_length == 0: t = 0 # handle zero segment lengths
                else: t = (target_distance - dist_cumulative[i - 1]) / segment_length
                resampled_point = p1 + t * (p2 - p1)
                resampled_points.append(resampled_point)
                break
    
    return np.array(resampled_points)

def line_length(resampled_points):
    """
    Calculate the total length of the resampled line.
    Args: resampled_points: The resampled points along the medial line.
    Returns:total_length: The total length of the resampled path.
    """
    total_length = 0.0
    for i in range(1, len(resampled_points)):
        dist = np.linalg.norm(resampled_points[i] - resampled_points[i - 1])
        total_length += dist
    return total_length

def create_skeleton(mask):
    """input mask, get initial, unfiltered, unlengthened skeleton"""
    mask = np.array(mask)
    skeleton = skeletonize(mask)
    return skeleton

def mask_area(mask):
    """input mask, get area (pixels)"""
    binary_mask = (mask > 0).astype(np.uint8) # convert to binary
    area = np.sum(binary_mask)
    return area

def get_cross_sectional_lengths(mask):  
    """input mask, get cross sectional lengths down the body,
    in 5% increments given the resampled medial line (which draws 20 points)"""
    medial_line_raw = compute_extended_path(create_skeleton(mask), mask, num_points_src=5, num_points_dst=5)[3] # medial line
    medial_line = resample_line(medial_line_raw, num_points=20) # pull medial line points (5% increments)

    directions = []
    for i in range(1, line_length(medial_line) - 1):
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

def get_cross_sectional_points(resampled_points, mask):
    """Get the two cross-sectional points at each of the resampled points along the medial line."""
    
    total_length = len(resampled_points)  # resampled points (20 points)
    cross_section_points = []
    for i in range(1, total_length - 1):  # skip the first and last points
        point = resampled_points[i]
        
        p1 = resampled_points[i - 1]
        p2 = resampled_points[i + 1]
        direction = np.array([p2[1] - p1[1], p1[0] - p2[0]])  # Perpendicular direction
        direction = direction / np.linalg.norm(direction)

        # Find the intersections in both directions (forward and backward)
        forward_intersection = find_intersection(mask, point, direction)
        backward_intersection = find_intersection(mask, point, -direction)
        if forward_intersection and backward_intersection:
            cross_section_points.append((forward_intersection, backward_intersection))
    
    return cross_section_points

def get_cross_sectional_percentiles_and_lengths(cross_section_points):
    """ Inputs: cross section points returned from get_cross_sectional_points().
    Returns two dictionaries:
    1. Mapping cross-sectional points to their percentiles.
    2. Mapping cross-sectional points to their lengths (distances)."""
    
    # use the first and last distances to determine head/tail orientation
    first_three_distances = [
        np.linalg.norm(np.array(cross_section_points[i][0]) - np.array(cross_section_points[i][1])) 
        for i in range(3)]  # First three points (5%, 10%, 15%)

    last_three_distances = [
        np.linalg.norm(np.array(cross_section_points[i][0]) - np.array(cross_section_points[i][1])) 
        for i in range(-3, 0)]  # Last three points (85%, 90%, 95%)
    
    if np.mean(first_three_distances) > np.mean(last_three_distances): head_is_top = True  # Head is at the top (first entries are larger)
    else: head_is_top = False  # Tail is at the top (last entries are larger)

    # initialize dictionaries
    cross_section_percentiles = {}
    cross_section_lengths = {}

    # map cross-sectional points and lengths to percentiles
    total_points = len(cross_section_points)
    for i, (forward, backward) in enumerate(cross_section_points):
        percentile = (i + 1) * 5 
        length = np.linalg.norm(np.array(forward)- np.array(backward))

        if head_is_top:
            cross_section_percentiles[(tuple(forward), tuple(backward))] = percentile # map as expected
            cross_section_lengths[(tuple(forward), tuple(backward))] = length
        else:
            cross_section_percentiles[(tuple(forward), tuple(backward))] = 100 - percentile # reverse the mapping
            cross_section_lengths[(tuple(forward), tuple(backward))] = length

    return cross_section_percentiles, cross_section_lengths

def get_ls_fs_ps(cross_section_points, mask):
    """Gets the points and lengths corresponding to the 25th (LS), 40th (FS), and 50th (PS) percentiles.
    Inputs: cross section points returned from get_cross_sectional_points().
    Returns: Dictionary with LS, FS, and PS mapped to the corresponding 'points' and 'length' """

    # get percentiles and lengths
    cross_section_percentiles, cross_section_lengths = get_cross_sectional_percentiles_and_lengths(cross_section_points)

    # initialize dictionary
    ls_fs_ps = {}
    percentiles_of_interest = [25, 40, 50]  # test new positions
    for percentile in percentiles_of_interest:
        found = False  # Flag to track if the percentile was found
        for (forward, backward), p in cross_section_percentiles.items():  # find cross sectional point
            if p == percentile:
                distance = cross_section_lengths[(tuple(forward), tuple(backward))]
                if percentile == 25:
                    ls_fs_ps["LS"] = {"points": (forward, backward), "length": distance}
                elif percentile == 40:
                    ls_fs_ps["FS"] = {"points": (forward, backward), "length": distance}
                elif percentile == 50:
                    ls_fs_ps["PS"] = {"points": (forward, backward), "length": distance}
                found = True
                break  # exit the loop once the percentile is found

        if not found: # if percentile not found, assign NA
            if percentile == 25:
                ls_fs_ps["LS"] = {"points": "NA", "length": "NA"}
            elif percentile == 40:
                ls_fs_ps["FS"] = {"points": "NA", "length": "NA"}
            elif percentile == 50:
                ls_fs_ps["PS"] = {"points": "NA", "length": "NA"}

    return ls_fs_ps

def compute_bodyspan(ls_fs_ps):
    """Given the dictionary returned from get_ls_fs_ps, return body span as the average of these three lengths.
    If any of the keys (LS, FS, PS) are missing or have 'NA', return 'NA'."""

    # check if any keys are NA
    if ls_fs_ps["LS"]["length"] == "NA" or ls_fs_ps["FS"]["length"] == "NA" or ls_fs_ps["PS"]["length"] == "NA": return "NA"
    
    # otherwise calculate length as usual
    average_length = (ls_fs_ps["LS"]["length"] + ls_fs_ps["FS"]["length"] + ls_fs_ps["PS"]["length"]) / 3

    return average_length

def get_section_areas(cross_section_points, mask):
    """
    Get true mask area (in pixels) between each consecutive pair of cross sections.
    
    Inputs: cross section points and shark mask
    Returns:section_areas: list of areas for each section (len = len(cross_section_points)-1)
    """
    # convert mask to biinary
    binary_mask = (mask > 0).astype(np.uint8) 

    # get mask contour (longest)
    contour = max(measure.find_contours(mask, 0.5), key=len)

    def nearest_idx(pt):
        return np.argmin(np.linalg.norm(contour - pt, axis=1))

    def contour_segment(start, end):
        return contour[start:end+1] if start <= end else np.vstack((contour[start:], contour[:end+1]))

    section_areas = []
    for i in range(len(cross_section_points) - 1):
        # unpack current & next cross section
        f1, b1 = cross_section_points[i]
        f2, b2 = cross_section_points[i+1]

        # find contour indices
        idx_f1, idx_f2 = nearest_idx(f1), nearest_idx(f2)
        idx_b1, idx_b2 = nearest_idx(b1), nearest_idx(b2)

        # edge on one side
        edge_f = contour_segment(idx_f1, idx_f2)
        # edge on the other side (reverse to close polygon properly)
        edge_b = contour_segment(idx_b2, idx_b1)[::-1]

        # build polygon and get pixels
        poly = np.vstack((edge_f, edge_b))
        rr, cc = sk_polygon(poly[:, 0], poly[:, 1], binary_mask.shape)

        # count actual binary_mask pixels inside polygon
        section_areas.append(np.sum(binary_mask[rr, cc]))

    return section_areas

import numpy as np

def get_headtail_area_and_length(section_areas):
    """
    Inputs: section_areas: areas from get_section_areas(), where each element corresponds to a 5% segment along the body
        and resampled_points: evenly spaced medial line points
    
    Returns: headtail_area: summed mask area between 20% and 70%.
        length_20_70: medial line length between 20% and 70%.
    """
    # Each section is 5%, so 20% starts at section index 4 (0-based)
    start_idx = 4
    end_idx = 13  # up to 70% (between 65% and 70%)

    headtail_area = sum(section_areas[start_idx:end_idx])

    return headtail_area

def compute_bai(headtail_area, total_length):
    """
    Compute the BAI (Body Area Index) as: BAI = [(SA) / (HT + TL)^2] * 100 (per Burnett et al 2018)

    Inputs: headtail_area (Area between 20% and 70% of the body), headtail_length (Medial line length between 20% and 70%),
        total_length (Total medial line length (0% to 100%))
    Returns: computed BAI value.
    """
    roi_proportion = 0.5
    denom = (roi_proportion*total_length)**2
    bai = (headtail_area / denom) * 100
    return bai

def compute_asc(fs, ls, ps, tl):
    """
    Compute the ASC (Span Condition Analysis) (per Irschick & Hammerschlag 2014)

    Inputs: fs, ls, ps, and tl
    Returns: computed ASC values
    """
        
    # convert TL to FL using Logan et al. 2018
    fl = 1.79 + 0.89*tl 

    # compute approximate ASC using Irschick & Hammerschlag 2014
    def d_to_halfc(d): return (0.5)*(np.pi)*d # convert diameter to 1/2 circumference for comparability 
    ckc = d_to_halfc(ps)/2 # estimate ckc circumference from Irschick & Hammerschlag 2014

    asc_skel = (d_to_halfc(fs) + d_to_halfc(ls) + d_to_halfc(ps) + ckc)/fl

    return asc_skel


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

#########################

def process_biometrics(root_predictions, pred_files, use_clean=True, use_extend=True): 
    """compute morphometric variables from input pred_files, returns dataframe.
        if use_clean=False, skips connect_clean_mask step
        if use_extend=False, skips center line pruning and extension"""
    data = []  # store rows for df
    interval_cols = [f"width_{i}_skel" for i in range(5, 100, 5)] # widths
    
    for file in pred_files:
        image_name = file.replace('pred_', '').replace('.png', '.JPG')
        mask_path = os.path.join(root_predictions, file) # full path 
        mask_raw = Image.open(mask_path)
        mask = np.array(mask_raw)

        # initialize widths as NaN
        widths_dict = {col: np.nan for col in interval_cols}

        # check for empty mask
        if not mask.any():
            print(f"{image_name}: empty mask")
            empty_row = [image_name] + [0]*10 + [np.nan]*(len(interval_cols)-1)
            data.append(empty_row)
            continue
        
        # optionally clean mask
        mask_cleaned = connect_clean_mask(mask) if use_clean else mask # connect tails, clean out artifacts
        mask_skeleton = create_skeleton(mask_cleaned) # pull base skeleton 

        # check for empty (or too small) skeleton 
        if np.count_nonzero(mask_skeleton) < 2:
            print(f"{image_name}: skeleton too small") 
            empty_row = [image_name] + [0]*10 + [np.nan]*(len(interval_cols)-1)
            data.append(empty_row)
            continue
        
        # optionally clean and extend skeleton
        if use_extend:
            skeleton_coords = compute_extended_path(mask_skeleton, mask_cleaned, num_points_src=5, num_points_dst=5)[3] # pull extended medial line
        else: # pull longest branch
            branch_data = summarize(Skeleton(mask_skeleton), separator="_")
            longest_branch_idx = branch_data['euclidean_distance'].idxmax()
            longest_branch = branch_data.iloc[longest_branch_idx]
            skeleton_coords = np.array(longest_branch[['image_coord_src_0','image_coord_src_1',
                                           'image_coord_dst_0','image_coord_dst_1']]).reshape(-1,2)

        skeleton_resampled = resample_line(skeleton_coords, num_points = 20) # resampled points to smooth

        if len(skeleton_resampled) < 2:
            print(f"{image_name}: resampling failed") 
            empty_row = [image_name] + [0]*10 + [np.nan]*(len(interval_cols)-1)  
            data.append(empty_row)
            continue

        skeleton_TL = line_length(skeleton_resampled) # extract medial tl
        body_area = mask_area(mask_cleaned) # extract body area
        cross_sectional_points = get_cross_sectional_points(skeleton_resampled, mask_cleaned) # extract cx points

        # check for invalid cross sections
        if len(cross_sectional_points) < 7:
            print(f"{image_name}: invalid cross sections") 
            empty_row = [image_name] + [0]*10 + [np.nan]*(len(interval_cols)-1)
            data.append(empty_row)
            continue
        
        # compute all major metrics
        section_areas = get_section_areas(cross_sectional_points, mask_cleaned)
        headtail_area = get_headtail_area_and_length(section_areas)
        skeleton_bai = compute_bai(headtail_area, skeleton_TL) # compute bai
        ls_fs_ps = get_ls_fs_ps(cross_sectional_points, mask_cleaned) # pull out the ls fs ps dictionary
        skeleton_FS = ls_fs_ps["FS"]["length"] # grab frontal span
        skeleton_LS = ls_fs_ps["LS"]["length"] # grab lateral span
        skeleton_PS = ls_fs_ps["PS"]["length"] # grab proximal span
        skeleton_BS = compute_bodyspan(ls_fs_ps) # compute the average body span
        skeleton_bsr = skeleton_BS/skeleton_TL # compute average bsr
        skeleton_asc = compute_asc(skeleton_FS, skeleton_LS, skeleton_PS, skeleton_TL) # compute asc 

        # pull all widths and name accordingly
        cross_section_percentiles, cross_section_lengths = get_cross_sectional_percentiles_and_lengths(cross_sectional_points)
        for (forward, backward), p in cross_section_percentiles.items():
            p_int = int(round(p))
            if 5 <= p < 100 and p % 5 == 0:
                col_name = f"width_{p}_skel"
                col_name = f"width_{p_int}_skel"
                widths_dict[col_name] = cross_section_lengths[(tuple(forward), tuple(backward))]
        row = [image_name, skeleton_TL, body_area, skeleton_FS, skeleton_LS, skeleton_PS,
               skeleton_BS, skeleton_bai, skeleton_asc, skeleton_bsr] + list(widths_dict.values())
        data.append(row)

    cols = ['filename', 'skeleton_TL', 'body_area', 'skeleton_FS', 'skeleton_LS', 'skeleton_PS',
                'skeleton_BS', 'BAI_skel', 'ASC_skel', 'BSR_skel'] + interval_cols
        
    df = pd.DataFrame(data, columns=cols)

    return(df)

def reconstruct_pixels_from_crop(df, crop_size, img_size, use_custom_crop):
    pixel_transf_factors = []
    for _, row in df.iterrows():
        relative_altitude = row['relative_altitude']  # change as needed
        img_width = row['image_width']  # change as needed
        img_height = row['image_height'] # change as needed

        if use_custom_crop:
            if pd.isna(relative_altitude) or pd.isna(img_width): # debugging
                print(f"Skipping row due to NaN: filename={row['filename']}, relative_altitude={relative_altitude}, img_width={img_width}")
                continue  # skip this iteration, go to next row
            crop_size = compute_custom_crop_size(relative_altitude, img_width) # compute custom
        else:
            crop_size = crop_size # pull input

        if crop_size == 0: #if no crop - need to pull original image size
            scale_width = img_width/img_size
            scale_height = img_height/img_size
            pixel_transf_factor = (scale_width+scale_height)/2 # applying average h/w transf - this is not comprehensive!

        else: 
            pixel_transf_factor = crop_size / img_size
        pixel_transf_factors.append(pixel_transf_factor)

    df['pixel_transf_factor'] = pixel_transf_factors

    return df

def compute_custom_crop_size(relative_altitude, img_width, base_crop_size=896):
    """
    Custom crop size based on relative altitude and image width
    Args:
        relative_altitude (float): The relative altitude of the image (in m).
        img_width (int): The width of the image in pixels.
        base_crop_size (int): The base crop size (default is 896).

    Returns:
        crop_size (int): The calculated crop size in pixels.
    """
    # if no relative altitude provided
    if relative_altitude is None:
        return base_crop_size
    
    # constants for camera - hard coded to P4A
    sensor_width_mm = 13.2  # mm
    focal_length_mm = 8.8   # mm
    shark_length_cm = 550   # cm (max shark length)
    shark_length_m = shark_length_cm / 100

    # calculate gsd (cm/pixel)
    gsd_cm_per_pixel = (sensor_width_mm * relative_altitude * 100) / (focal_length_mm * img_width)

    # calculate shark length (pixels)
    pixels_shark = shark_length_m * 100 / gsd_cm_per_pixel

    # crop size as minimum possible crop given max shark size
    crop_size = math.ceil(pixels_shark / 112) * 112

    # if crop size is bigger than ImW, round down
    if crop_size > img_width:
        crop_size = (img_width // 112) * 112
        
    return crop_size

def compute_calibrations(df):
    """
    Adds altitude offsets to calibration dataframe 

    Input: calibration points dataframe with columns 'relative_altitude' (m), 'image_width' (pix), 
        calibration_length_pixels and 'true_length_cm'
    Ouput: df with computed GSD, true GSD, back-calculated altitude, and altitude offset on a per-image basis
    """
    # camera specifications (P4A) ## hard coded to Phantom 4 Advanced
    sensor_width_mm = 13.2
    focal_length_mm = 8.8

    # filter out calibration images with a shallow gimbal angle 
    gimbal_angle_threshold = -80
    df = df[df['gimbal_pitch_deg'] <= gimbal_angle_threshold].copy() 

    # add flight_date variable
    if df['date_time'].dtype == 'O':
        df['date_time'] = pd.to_datetime(
            df['date_time'].str.replace(r'^(\d{4}):(\d{2}):(\d{2})', r'\1-\2-\3', regex=True),
            format='%Y-%m-%d %H:%M:%S')
    df['flight_date'] = 'flight' + df['flight'].astype(str) + '-' + df['date_time'].dt.date.astype(str)

    # calculate observed gsd (relative to calibration object)
    df['gsd_cm_per_pixel'] = ((df['relative_altitude'] - df['distance_off_surface_cm']/100) * sensor_width_mm * 100)/(focal_length_mm * df['image_width'])

    # calculate observed photogrammetric length (cm)
    df['photog_length_cm'] = df['length_pixels'] * df['gsd_cm_per_pixel']

    # back-calculate true gsd in the plane of the calibration object
    df['true_gsd_cm_per_pixel'] = df['true_length_cm'] / df['length_pixels']

    # back-calculate true altitude (m) to the sea surface; accounting for plane differences
    df['true_altitude_m'] = (df['true_gsd_cm_per_pixel'] * focal_length_mm * df['image_width'])/(sensor_width_mm * 100) + df['distance_off_surface_cm']/100
    
    # calculate altitude difference 
    df['altitude_diff_m'] = df['relative_altitude'] - df['true_altitude_m']

    # calculate percent errors
    df['percent_length_error'] = np.abs(df['photog_length_cm'] - df['true_length_cm']) / df['true_length_cm'] * 100
    df['signed_percent_length_error'] = (df['photog_length_cm'] - df['true_length_cm']) / df['true_length_cm'] * 100

    return df

def compute_group_calibrations(df):
    """
    Summarizes mean altitude difference based on individual image values (using compute_calibrations) 
    contained in calibration metadata for flight-date groups

    Input: calibration points dataframe 
    Output: df with one row per flight_date, adding 'mean_altitude_diff_m', 'var_altitude_diff_m', 'num_calib_images'
    """

    # compute per-image calibration values
    df = compute_calibrations(df)

    # create date-flight groupings
    if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['date_time'] = pd.to_datetime(df['date_time'], format='%Y:%m:%d %H:%M:%S')

    df['date'] = df['date_time'].dt.date.astype(str)
    df['flight_date'] = 'flight' + df['flight'].astype(str) + '-' + df['date']

    # aggregate across date-flight groups
    calib_corrections = df.groupby('flight_date').agg(
        mean_altitude_diff_m=('altitude_diff_m', 'mean'),
        var_altitude_diff_m=('altitude_diff_m', 'var'),
        num_calib_images=('altitude_diff_m', 'count')
    ).reset_index()

    return calib_corrections

def apply_altitude_calibration(df, calib_df):
    """
    Applies altitude calibration from calib_df (processed via compute_calibrations) 
    to each unique flight-date group in shark images
    
    Input: 
      df: shark image df
      calib_df: calibration df with correction factors per flight_date, processed via compute_calibrations 
      (has  'flight_date', 'mean_altitude_diff_m', 'var_altitude_diff_m', 'num_calib_images' cols)
    
    Output: df with 'corrected_relative_altitude', 'mean_altitude_diff_m','var_altitude_diff_m', and 'num_calib_images'
    """
    # convert datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['date_time'] = pd.to_datetime(df['date_time'], format='%Y:%m:%d %H:%M:%S')

    # create date-flight group
    df['date'] = df['date_time'].dt.date.astype(str)
    df['flight_date'] = 'flight' + df['flight'].astype(str) + '-' + df['date']

    # merge by flight-date
    df = df.merge(
        calib_df[['flight_date', 'mean_altitude_diff_m', 'var_altitude_diff_m', 'num_calib_images']],
        on='flight_date', how='left')

    # apply correction
    df['corrected_relative_altitude'] = df['relative_altitude'] - df['mean_altitude_diff_m']

    # NA calibration for shark images with a shallow gimbal angle 
    gimbal_angle_threshold = -80
    df.loc[df['gimbal_pitch_deg'] > gimbal_angle_threshold, 'corrected_relative_altitude'] = np.nan

    return df

def photogrammetric_conversion(df):
    """
    Converts pixel measurements back to original image sizes before cropping and 
    converts pixels to cm using photogrammetry
    
    Requires: 'corrected_relative_altitude' (pre-processed by apply_altitude_calibration()), 
    'image_width', 'pixel_transf_factor' (pre-processed with reconstruct_pixels_from_crop()),
    'skeleton_TL' and 'skeleton_BS', 'width_*' columns (pre-processed with process_biometrics())
    """
    # replace string nans 
    df = df.copy()
    df = df.replace(["NA", "NaN", "nan", "null", ""], np.nan)

    # hard-coded camera parameters (Phantom 4 Advanced)
    sensor_width_mm = 13.2
    focal_length_mm = 8.8

    # hard-coded depth parameter (approx depth of shark below sea surface)
    body_depth = 1

    # compute GSD using corrected altitude, factoring in body depth
    df['GSD_cm'] = ((df['corrected_relative_altitude'] + body_depth) * sensor_width_mm * 100) / (focal_length_mm * df['image_width'])

    # re-compute pixel lengths before cropping, apply gsd
    main_measurements = ['TL', 'BS', 'FS', 'LS', 'PS']
    for m in main_measurements:
        df[f'{m}_skel_pixels'] = df[f'skeleton_{m}'] * df['pixel_transf_factor'] # convert to original px
        df[f'{m}_skel_cm'] = df[f'{m}_skel_pixels'] * df['GSD_cm'] # convert to cm using gsd

    # apply for width columns
    width_cols = [col for col in df.columns if col.startswith('width_')]
    for col in width_cols:
        df[col + '_pixels'] = df[col] * df['pixel_transf_factor']
        df[col + '_cm'] = df[col + '_pixels'] * df['GSD_cm']

    # convert ground-truth measurements to centimeters using GSD
    df['TL_cm'] = df['TL_pixels'] * df['GSD_cm']
    df['BS_pixels'] = (df['LateralSpan_Pixels'] + df['FrontalSpan_Pixels'] + df['ProximalSpan_Pixels']) / 3
    df['BS_cm'] = df['BS_pixels'] * df['GSD_cm']

    return df


## plotting functions ##################################################################################################

def plot_extended_path(extended_path, shortest_path, src_coords, dst_coords, img):
    """
    Plots the extended path along with the original skeleton, source, and destination.
    Args:
    - extended_path: The extended path to plot.
    - shortest_path: The original shortest path.
    - src_coords: Source coordinates.
    - dst_coords: Destination coordinates.
    - img: The image (binary mask) to plot.
    """
    shortest_path = np.array(shortest_path)

    plt.imshow(img, cmap='gray')  # Display image/mask
    plt.plot(extended_path[:, 1], extended_path[:, 0], color='yellow', label='Extended Skeleton')
    plt.scatter(shortest_path[:, 1], shortest_path[:, 0], color='red', label='Skeleton points', s=8)
    plt.scatter(src_coords[1], src_coords[0], color='blue', label='Source', s=11)
    plt.scatter(dst_coords[1], dst_coords[0], color='green', label='Destination', s=11)
    plt.legend()
    plt.axis('off')
    return plt.gcf()


def plot_cross_sections(img, cross_section_points):
    """Plot the cross sections across the body and highlight the 25th, 40th, and 50th percentile lines in blue."""
    plt.imshow(img, cmap='gray')
    percentiles_positions = {
        25: 5,    # 25th percentile corresponds to the 5th position 
        40: 8,    # 40th percentile corresponds to the 8th position
        50: 10     # 50th percentile corresponds to the 10th position
    }
    for i, (forward, backward) in enumerate(cross_section_points):
        if i == percentiles_positions[25] - 1:  # 0-based index
            color = 'blue'  # Highlight the 25th percentile line
        elif i == percentiles_positions[40] - 1:  # 0-based index
            color = 'blue'  # Highlight the 40th percentile line
        elif i == percentiles_positions[50] - 1:  # 0-based index
            color = 'blue'  # Highlight the 50th percentile line
        else:
            color = 'red'  # Default color for other lines
        plt.plot([forward[1], backward[1]], [forward[0], backward[0]], color=color, alpha=0.5)

    plt.title("Cross-Sectional Lines from Resampled Line")
    plt.show()

    
def plot_widest_cross_section(mask, cross_section_points):
    """plot only the widest cross secton along the body"""
    # Find the widest cross section
    ###########widest_pair, max_distance = get_widest_cross_section(cross_section_points) - #get widest cross section no longer exists

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

def plot_mask_with_skeleton_and_cross_sections(img, mask):
    """plot the mask with cx sections on the left, image with cx sections on the right"""

    mask_cleaned = connect_clean_mask(mask) # connect tails, clean out artifacts
    mask_skeleton = create_skeleton(mask_cleaned)

    # handle empty or tiny skeleton
    if not np.any(mask_cleaned) or np.sum(mask_skeleton) < 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor('black')
        for ax in axes:
            ax.set_facecolor('black')
            ax.axis('off')
        return fig
    
    # process skeleton
    skeleton_extended = compute_extended_path(mask_skeleton, mask_cleaned, num_points_src=5, num_points_dst=5)[3] # pull extended medial line
    skeleton_resampled = resample_line(skeleton_extended) # resampled points to smooth

    # widths
    cross_sectional_points = get_cross_sectional_points(skeleton_resampled, mask_cleaned) # extract cx points
    ls_fs_ps = get_ls_fs_ps(cross_sectional_points, mask)

    # plotting two panels side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    fig.patch.set_facecolor('black')  # Set the entire figure background to black

    # plot mask and morphometrics
    ax1 = axes[0]
    ax1.set_facecolor('black')  # Set the background of the axes to black
    ax1.imshow(mask, cmap='gray', vmin=0, vmax=255, alpha=0.5)

    for y, x in skeleton_resampled:
        ax1.plot(x, y, 'ro', markersize=1, alpha=0.8)  # Plot each skeleton point as a red dot

    # Plot LS (25th percentile)
    try: # functionality in case LS is missing
        ls_points = ls_fs_ps.get("LS", {}).get("points", None)
        if ls_points:
            forward, backward = ls_points
            ax1.plot([forward[1], backward[1]], [forward[0], backward[0]], 'r-', linewidth=2)
            ax1.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='red', s = 10, zorder=5)
    except Exception as e: print(f"Error plotting LS: {e}")

    # Plot FS (40th percentile)
    fs_points = ls_fs_ps.get("FS", {}).get("points", None)
    try: # functionality in case FS is missing
        if fs_points:
            forward, backward = fs_points
            ax1.plot([forward[1], backward[1]], [forward[0], backward[0]], 'r-', linewidth=2)
            ax1.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='red', s = 10, zorder=5)
    except Exception as e: print(f"Error plotting FS: {e}")

    # Plot PS (50th percentile)
    ps_points = ls_fs_ps.get("PS", {}).get("points", None)
    try: # functionality in case PS is missing
        if ps_points:
            forward, backward = ps_points
            ax1.plot([forward[1], backward[1]], [forward[0], backward[0]], 'r-', linewidth=2)
            ax1.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='red', s = 10, zorder=5)
    except Exception as e: print(f"Error plotting PS: {e}")

    ax1.set_title('Morphometrics Mask', color='white')  # Title in white for visibility
    ax1.axis('off')  # Hide axis

    # Plot original image
    ax2 = axes[1]
    ax2.set_facecolor('black')
    ax2.imshow(img, cmap='gray', vmin=0, vmax=255)

    # Plot LS (25th percentile)
    try: 
        if ls_points:
            forward, backward = ls_points
            ax2.plot([forward[1], backward[1]], [forward[0], backward[0]], 'b-', linewidth=2, label='LS (25%)')
            ax2.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='blue', s = 10, zorder=5)
    except Exception as e: print(f"Error plotting LS: {e}")

    # Plot FS (40th percentile)
    try:
        if fs_points:
            forward, backward = fs_points
            ax2.plot([forward[1], backward[1]], [forward[0], backward[0]], 'g-', linewidth=2, label='FS (40%)')
            ax2.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='green', s = 10, zorder=5)
    except Exception as e: print(f"Error plotting FS: {e}")

    # Plot PS (50th percentile)
    try:
        if ps_points:
            forward, backward = ps_points
            ax2.plot([forward[1], backward[1]], [forward[0], backward[0]], 'r-', linewidth=2, label='PS (50%)')
            ax2.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='red', s = 10, zorder=5)
    except Exception as e: print(f"Error plotting LS: {e}")
    
    ax2.set_title('Original Image w/Spans', color='white')  # Title for the original image
    ax2.axis('off')  # Hide axis

    # Add legend
    ax1.legend(['LS', 'FS', 'PS'])

    plt.tight_layout()
    return fig  

def plot_mask_with_all_cross_sections(mask):
    """
    Plot the mask with all cross sections along the body."""
    import matplotlib.pyplot as plt
    import numpy as np

    mask_cleaned = connect_clean_mask(mask)
    mask_skeleton = create_skeleton(mask_cleaned)

    # handle empty or tiny skeleton
    if not np.any(mask_cleaned) or np.sum(mask_skeleton) < 2:
        plt.figure(figsize=(6,6))
        plt.imshow(mask, cmap='gray', vmin=0, vmax=255, alpha=0.5)
        plt.title('Empty or invalid mask')
        plt.axis('off')
        return

    # process skeleton
    skeleton_extended = compute_extended_path(mask_skeleton, mask_cleaned, num_points_src=5, num_points_dst=5)[3]
    skeleton_resampled = resample_line(skeleton_extended)

    # get cross-sectional points
    cross_sections = get_cross_sectional_points(skeleton_resampled, mask_cleaned)
    cross_percentiles, cross_lengths = get_cross_sectional_percentiles_and_lengths(cross_sections)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray', vmin=0, vmax=255, alpha=0.5)
    
    for (forward, backward), percentile in cross_percentiles.items():
        y_coords = [forward[0], backward[0]]
        x_coords = [forward[1], backward[1]]
        plt.plot(x_coords, y_coords, 'r-', linewidth=1)

        # label near the middle
        mid_y = np.mean(y_coords)
        mid_x = np.mean(x_coords)
        plt.text(mid_x, mid_y, f"{percentile}%", color='blue', fontsize=10, ha='center', va='center')

    plt.title('Mask with all cross sections')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_resampled_points_with_extended_path(img, mask):
    """
    Plot the resampled points used to calculate medial line
    Args: image and mask
    Returns: printed plot of resampled points
    """
    src_coords, dst_coords, shortest_path, extended_path = compute_extended_path(create_skeleton(mask), mask, num_points_src=5, num_points_dst=5)
    resampled_points = resample_line(extended_path)
    img_copy = np.copy(img)
    
    plt.imshow(img_copy, cmap='gray')  # Display image/mask
    plt.scatter(np.array(resampled_points)[:, 1], np.array(resampled_points)[:, 0], color='red', label='Resampled Points', s=3)    
    plt.legend()
    plt.axis('off')
    plt.show()


def plot_ls_fs_ps(cross_section_points, mask, img):
    """Plot the LS, FS, and PS lines on the original image."""
    ls_fs_ps = get_ls_fs_ps(cross_section_points, mask)
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')  # Plot the original mask (image)
    
    # LS (25th percentile)
    ls_points = ls_fs_ps.get("LS", {}).get("points", None) # LS (25th)
    if ls_points:
        forward, backward = ls_points
        plt.plot([forward[1], backward[1]], [forward[0], backward[0]], 'b-', linewidth=2, label='LS (25%)')
        plt.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='blue', s=10, zorder=5)

    # FS (40th percentile)
    fs_points = ls_fs_ps.get("FS", {}).get("points", None) # FS (40th)
    if fs_points:
        forward, backward = fs_points
        plt.plot([forward[1], backward[1]], [forward[0], backward[0]], 'g-', linewidth=2, label='FS (40%)')
        plt.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='green', s=10, zorder=5)

    # PS (50th percentile)
    ps_points = ls_fs_ps.get("PS", {}).get("points", None) # PS (50th)
    if ps_points:
        forward, backward = ps_points
        plt.plot([forward[1], backward[1]], [forward[0], backward[0]], 'r-', linewidth=2, label='PS (50%)')
        plt.scatter([forward[1], backward[1]], [forward[0], backward[0]], color='red', s=10, zorder=5)

    plt.title("Cross-Sectional Lines (LS, FS, PS)")
    plt.legend()
    plt.show()


def plot_image_mask_skeleton_cross_sections(img, mask, skeletonized_line, cross_section_points, resampled_points):
    """
    Plots four images side by side:
    1. Original Image, 2. Predicted Mask, 3. Skeletonized Line, 4. Cross Sections (LS, FS, PS)
    
    Args:
    - img: The original image.
    - mask: The predicted binary mask.
    - skeletonized_line: The skeletonized version of the mask.
    - cross_section_points: The points for LS, FS, and PS cross-sections.
    """
    # Get LS, FS, and PS lines
    ls_fs_ps = get_ls_fs_ps(cross_section_points, mask)

    # Create the figure with 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot 1: Original Image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot 2: Predicted Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    # Plot 3: Mask with Skeletonized Line
    axes[2].imshow(mask, cmap='gray')  # Background: Predicted Mask
    axes[2].set_title('Skeletonized Line')
    axes[2].axis('off')
    skeleton_color = mcolors.to_rgba('red', alpha=0.7)  # RGBA color with transparency
    axes[2].imshow(skeletonized_line, cmap='gray', alpha=0)  # Make sure it's not drawn in gray
    axes[2].contour(skeletonized_line, colors=[skeleton_color], linewidths=1)
    
    # Overlay the resampled points in red in the third panel
    for point in resampled_points:
        axes[2].scatter(point[1], point[0], color='red', s=5, zorder=5)

    # Plot 4: Mask with Cross Sections (LS, FS, PS) and Skeletonized Line
    axes[3].imshow(mask, cmap='gray')  # Background: Predicted Mask
    axes[3].set_title('Cross Sections')
    axes[3].axis('off')

    # LS (25th percentile)
    ls_points = ls_fs_ps.get("LS", {}).get("points", None)
    if ls_points:
        forward, backward = ls_points
        axes[3].plot([forward[1], backward[1]], [forward[0], backward[0]], 'b-', linewidth=2, label='LS (25%)')
        axes[3].scatter([forward[1], backward[1]], [forward[0], backward[0]], color='blue', s=10, zorder=5)

    # FS (40th percentile)
    fs_points = ls_fs_ps.get("FS", {}).get("points", None)
    if fs_points:
        forward, backward = fs_points
        axes[3].plot([forward[1], backward[1]], [forward[0], backward[0]], 'g-', linewidth=2, label='FS (40%)')
        axes[3].scatter([forward[1], backward[1]], [forward[0], backward[0]], color='green', s=10, zorder=5)

    # PS (50th percentile)
    ps_points = ls_fs_ps.get("PS", {}).get("points", None)
    if ps_points:
        forward, backward = ps_points
        axes[3].plot([forward[1], backward[1]], [forward[0], backward[0]], 'y-', linewidth=2, label='PS (50%)')
        axes[3].scatter([forward[1], backward[1]], [forward[0], backward[0]], color='yellow', s=10, zorder=5)

    # Overlay the resampled points in red in the fourth panel
    for point in resampled_points:
        axes[3].scatter(point[1], point[0], color='red', s=5, zorder=5)  # (y, x) for image coordinates

    # Overlay the skeletonized line on top of the cross-sections (in the fourth panel)
    axes[3].contour(skeletonized_line, colors=[skeleton_color], linewidths=1)  # Overlay skeleton with the same color

    # Add legend to the last plot (cross-sections)
    axes[3].legend(loc='best')

    # Hide axes for all subplots for cleaner look
    for ax in axes:
        ax.axis('off')

    # Adjust layout to avoid overlap and show the plot
    plt.tight_layout()
    plt.show()