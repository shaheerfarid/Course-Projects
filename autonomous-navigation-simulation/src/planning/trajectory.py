

import numpy as np

def get_curvature_and_direction(poly):
  """
  Estimates the curvature and turn direction from the averaged (middle) polynomial
  Args:
    poly: the middle polynomial
  Returns:
    A tuple containing the curvature and the strings "left" or "right"
  """
  # evalute the curvature and turn direction at this point
  pt = poly.domain.mean()

  # k(poly)(x) computes the curvature of a polynomial 'poly' at the point 'x', using the formula curvature = |p''| / (1+(p')^2)^1.5
  k = lambda poly: lambda x: np.abs(poly.deriv(m=2)(x)) / ((1 + poly.deriv(m=1)(x) ** 2) ** 1.5)

  # positive second derivative => turn right, negative second derivative => turn left
  twice_deriv = poly.deriv(m=2)(pt)
  direction = "right" if twice_deriv > 0 else "left"

  return k(poly)(pt), direction

