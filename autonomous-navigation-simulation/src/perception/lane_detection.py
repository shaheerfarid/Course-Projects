"""
Lane Detection Module

This module implements a classical computer vision pipeline for detecting
lane markings from RGB images using brightness thresholding and spatial tracking.
"""

import numpy as np

def get_bright_pixels(img):
    """
    Identifies bright pixels in an RGB image using a fixed intensity threshold.

    Args:
        image (np.ndarray): RGB image of shape (H, W, 3).
        threshold (int): Pixel intensity threshold for brightness detection.

    Returns:
        np.ndarray: Boolean mask of shape (H, W) where True indicates bright pixels.
    """
    return (img > 175).all(axis=2)


def get_lane_beginnings(mask):
    """
    Estimates the horizontal positions of the left and right lane bases
    using a histogram-based approach on the lower portion of the image.
        
    Args:
        mask (np.ndarray): Boolean mask of bright pixels, shape (H, W).

    Returns:
        tuple[int, int]: Estimated x-coordinates of the left and right lanes.
    """

    height, width = mask.shape

    # Focus on bottom region where lanes are most visible
    bottom_mask = mask[int(0.95*height):]
    histogram = np.sum(bottom_mask, axis = 0)

    # Select strong lane responses
    tall_points_x = np.argwhere(histogram > np.max(histogram) / 3)
    bunch_boundary = np.mean(tall_points_x)
    return int(np.median(tall_points_x[tall_points_x < bunch_boundary])), int(np.median(tall_points_x[tall_points_x > bunch_boundary]))


def get_whole_lanes(mask, left_lane, right_lane):
   
    """
    Uses the beginning of the lanes to refine mask so that those unrelated bright pixels are filtered out.
    Args:
        mask: a NumPy array of shape (height, width) where True indicates the bright pixels
        left_lane: the x-coordinate of the beginning of the left lane
        right_lane: the x-coordinate of the beginning of the right lane
    Returns:
        A tuple of 2 boolean NumPy arrays, each array with shape (height, width), corresponding to the left lane and right lane.
    """

    window_width, window_height = 220, 72
    height, width = mask.shape

    rtn = [np.full((height, width), False) for _ in range(2)]
    for idx, lane in enumerate((left_lane, right_lane)):
        bx, by = lane - (window_width // 2), height - window_height

        for _ in range(9):
            bounding_box = mask[by:by+window_height, bx:bx+window_width]
            rtn[idx][by:by+window_height, bx:bx+ window_width] = bounding_box

            if not np.any(bounding_box):
                break

            x_values = np.broadcast_to(np.arange(bx, bx+window_width)[np.newaxis, :], (window_height, window_width))
            bx = int(np.mean(x_values[bounding_box])) - (window_width // 2)
            by = by - window_height

    return tuple(rtn)


