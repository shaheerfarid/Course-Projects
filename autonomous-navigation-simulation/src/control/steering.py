import numpy as np
from planning.trajectory import get_curvature_and_direction
from planning.polynomial_fit import get_middle_quadratic, get_quadratic_fit
from perception.lane_edges import extract_lane_edges


def get_direction_vector(curv, direction):
    """Finds the direction vector to move in"""
    r = 1 / curv  # find radius of tangent circle
    s = 350  # move 350 pixels along the tangent circle
    if direction == "left":
        dx = -r + r * np.cos(s / r)
        dy = -r * np.sin(s / r)
    elif direction == "right":
        dx = +r - r * np.cos(s / r)
        dy = -r * np.sin(s / r)
    return dx, dy


def extract_direction_vector(img):
    """
    Finds how much to move in the x and y direction.
    Args:
      img: a NumPy array of shape (height, width, 3) storing an image
    Returns:
      a tuple containing 2 floating point numbers.  The first one is dx, the second one is dy.
    """

    left_lane_edge, right_lane_edge = extract_lane_edges(img)
    left_poly, right_poly = get_quadratic_fit(left_lane_edge, right_lane_edge)
    middle_poly = get_middle_quadratic(left_poly, right_poly)
    curv, direction = get_curvature_and_direction(middle_poly)
    return get_direction_vector(curv, direction)
