"""
Lane Edge Extraction

This module extracts lane edges from detected lane pixels
using morphological operations and convolution filters.
"""

import numpy as np
import cv2 as cv
from perception.lane_detection import (
    get_bright_pixels,
    get_lane_beginnings,
    get_whole_lanes,
)


def get_lane_edges(left_lane, right_lane):
    """
    Extracts lane edges.
    Args:
      left_lane: a boolean NumPy array of shape (height, width) where True indicates the pixels for the left lane
      right_lane: a boolean NumPy array of shape (height, width) where True indicates the pixels for the right lane
    Returns:
      A tuple of 2 boolean NumPy arrays each with shape (height, width), containing the edges of the left lane and the right lane respectively
    """

    # define the kernels
    edge_left_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    edge_right_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    structuring_element = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )  # use this for erosion and closing

    # left lane erosion, closing and edge detection
    left_lane = cv.erode(left_lane.astype(np.uint8), structuring_element, iterations=3)
    left_lane = cv.morphologyEx(
        left_lane, cv.MORPH_CLOSE, structuring_element, iterations=10
    )
    left_lane = cv.filter2D(
        left_lane, ddepth=-1, kernel=edge_left_kernel
    ) | cv.filter2D(left_lane, ddepth=-1, kernel=edge_right_kernel)

    # right lane erosion, closing and edge detection
    right_lane = cv.erode(
        right_lane.astype(np.uint8), structuring_element, iterations=3
    )
    right_lane = cv.morphologyEx(
        right_lane, cv.MORPH_CLOSE, structuring_element, iterations=10
    )
    right_lane = cv.filter2D(
        right_lane, ddepth=-1, kernel=edge_left_kernel
    ) | cv.filter2D(right_lane, ddepth=-1, kernel=edge_right_kernel)

    return left_lane > 0, right_lane > 0


def extract_lane_edges(img):
    """
    The main function of section 1.  Extracts the lane edges from an image.
    Args:
      img: a NumPy array of shape (height, width, 3) storing an image
    Returns:
      A tuple of 2 boolean NumPy arrays each with shape (height, width), containing the edges of the left lane and the right lane respectively
    """

    mask = get_bright_pixels(img)
    left_lane_begin, right_lane_begin = get_lane_beginnings(mask)
    left_lane_whole, right_lane_whole = get_whole_lanes(
        mask, left_lane_begin, right_lane_begin
    )  # [[y, x], [y, x], ...]
    return get_lane_edges(left_lane_whole, right_lane_whole)
