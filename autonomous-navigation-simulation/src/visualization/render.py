import numpy as np
from control.steering import extract_direction_vector
from planning.polynomial_fit import get_middle_quadratic, get_quadratic_fit
from perception.lane_edges import extract_lane_edges
import matplotlib.pyplot as plt


def show_polynomials_and_vector(img):
    """
    Plots polynomials and the direction vector onto the image.
    """
    h, _, _ = img.shape

    left_poly, right_poly = get_quadratic_fit(*extract_lane_edges(img))
    middle_poly = get_middle_quadratic(left_poly, right_poly)

    dx, dy = extract_direction_vector(img)
    direction = "left" if dx < 0 else "right"

    # get the points to evaluate the curvature at
    pt = middle_poly.domain.mean()

    # get the xy coordinates of the polynomial at various points
    left_y, left_x = left_poly.linspace(n=50)
    middle_y, middle_x = middle_poly.linspace(n=50)
    right_y, right_x = right_poly.linspace(n=50)

    # load image into matplotlib
    plt.imshow(img)

    # plot polynomial points on top of the image
    plt.scatter(x=left_x, y=left_y, color="green", s=7)  # plot left poly
    plt.scatter(x=middle_x, y=middle_y, color="yellow", s=7)  # plot middle poly
    plt.scatter(x=right_x, y=right_y, color="blue", s=7)  # plot right poly
    plt.scatter(
        x=[middle_poly(pt)], y=[pt], color="purple"
    )  # plot point to evaluate curvature at
    plt.arrow(
        x=middle_poly(pt), y=pt, dx=dx, dy=dy, width=20, color=(1, 0, 0)
    )  # draw direction vector
    plt.text(
        x=20,
        y=h - 20,
        s=f"{direction} by {np.abs(dx)} px",
        fontsize="xx-large",
        color="white",
    )

    plt.show()
