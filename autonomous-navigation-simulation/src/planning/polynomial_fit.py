import numpy as np

def get_quadratic_fit(left_lane_edge, right_lane_edge):
  """
  Fits a quadratic to the left lane and right lane using the edges of the lane markings.
  Args:
    left_lane-edge: a boolean NumPy array of shape (height, width) where True indicates the pixels for the left lane edges
    right_lane_edge: a boolean NumPy array of shape (height, width) where True indicates the pixels for the right lane edges
  Returns:
    a tuple containing two quadratics.
    return_val[0] is the fitted left quadratic, and return_val[2] is the fitted right quadratic
  """

  # generate an appropriate coordinate grid
  h, w = left_lane_edge.shape
  grid = np.indices((h, w)).transpose(1, 2, 0)

  # NumPy slicing and indexing to pass the correct coordinates of edges to fit the polynomial
  left_poly = np.polynomial.polynomial.Polynomial.fit(x=grid[left_lane_edge][:, 0], y=grid[left_lane_edge][:, 1], deg=2)
  right_poly = np.polynomial.polynomial.Polynomial.fit(x=grid[right_lane_edge][:, 0], y=grid[right_lane_edge][:, 1], deg=2)

  return left_poly, right_poly


def get_middle_quadratic(left_poly, right_poly):
  """ Returns the average of the left and right polynomial """
  # average left and right quadratic, and handle NumPy domain issues
  middle_poly = (left_poly.convert(domain=[0, 1439]) + right_poly.convert(domain=[0, 1439])) / 2
  a, b, c, d = left_poly.domain[0], right_poly.domain[0], left_poly.domain[1], right_poly.domain[1]
  middle_poly = middle_poly.convert(domain=[min(a, b, c, d), max(a, b, c, d)])
  return middle_poly

