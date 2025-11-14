# -*- coding: utf-8 -*-

# import libraries.  This contains all the libraries you will need to complete this PA.
import numpy as np
import cv2 as cv
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
import os

# not used for any graded tasks.  Only for you to debug and visualize.
import matplotlib.pyplot as plt # plot images

# connect your Google drive to load the videos and images
if __name__ == "__main__":
  from google.colab import drive
  drive.mount('/content/drive')


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 2.1:  get_bright_pixels 	#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def get_bright_pixels(img):
  """
  Finds the location of bright pixels in an image.
  Args:
    img: a NumPy array of shape (height, width, 3) storing an image
  Returns:
    A boolean NumPy array of shape (height, width) where True indicates the location of bright pixels.
  """
  ### TODO: Your code goes here! ###

  return (img > 175).all(axis=2)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 2.2:  get_lane_beginnings 	#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def get_lane_beginnings(mask):
  """
  Estimates the x-coordinates of the lane beginnings.
  Args:
    mask: a boolean NumPy array of shape (height, width) where True indicates the location of bright pixels.
  Returns:
    A tuple of two integers, containing the estimated x-coordinate of the beginning of the left and right lane respectively.
  """
  ### TODO: Your code goes here! ###

  height, width = mask.shape
  bottom_mask = mask[int(0.95*height):]
  histogram = np.sum(bottom_mask, axis = 0)
  tall_points_x = np.argwhere(histogram > np.max(histogram) / 3)
  bunch_boundary = np.mean(tall_points_x)
  return int(np.median(tall_points_x[tall_points_x < bunch_boundary])), int(np.median(tall_points_x[tall_points_x > bunch_boundary]))


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 2.3:  get_whole_lanes 	#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

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
  ### Your code goes here! ###

  bw, bh = 220, 72
  height, width = mask.shape

  rtn = [np.full((height, width), False) for _ in range(2)]
  for idx, lane in enumerate((left_lane, right_lane)):
    bx, by = lane - (bw // 2), height - bh

    for _ in range(9):
      bounding_box = mask[by:by+bh, bx:bx+bw]
      rtn[idx][by:by+bh, bx:bx+bw] = bounding_box

      if not np.any(bounding_box):
        break

      x_values = np.broadcast_to(np.arange(bx, bx+bw)[np.newaxis, :], (bh, bw))
      bx = int(np.mean(x_values[bounding_box])) - (bw // 2)
      by = by - bh

  return tuple(rtn)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 2.4:  get_lane_edges 	#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def get_lane_edges(left_lane, right_lane):
  """
  Extracts lane edges.
  Args:
    left_lane: a boolean NumPy array of shape (height, width) where True indicates the pixels for the left lane
    right_lane: a boolean NumPy array of shape (height, width) where True indicates the pixels for the right lane
  Returns:
    A tuple of 2 boolean NumPy arrays each with shape (height, width), containing the edges of the left lane and the right lane respectively
  """
  ### Your code goes here! ###
  # define the kernels
  edge_left_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
  edge_right_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
  structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8) # use this for erosion and closing

  # left lane erosion, closing and edge detection
  left_lane = cv.erode(left_lane.astype(np.uint8), structuring_element, iterations=3)
  left_lane = cv.morphologyEx(left_lane, cv.MORPH_CLOSE, structuring_element, iterations=10)
  left_lane = cv.filter2D(left_lane, ddepth=-1, kernel=edge_left_kernel) | cv.filter2D(left_lane, ddepth=-1, kernel=edge_right_kernel)

  # right lane erosion, closing and edge detection
  right_lane = cv.erode(right_lane.astype(np.uint8), structuring_element, iterations=3)
  right_lane = cv.morphologyEx(right_lane, cv.MORPH_CLOSE, structuring_element, iterations=10)
  right_lane = cv.filter2D(right_lane, ddepth=-1, kernel=edge_left_kernel) | cv.filter2D(right_lane, ddepth=-1, kernel=edge_right_kernel)

  return left_lane == 1, right_lane == 1


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 2.5:  extract_lane_edges 	#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def extract_lane_edges(img):
  """
  The main function of section 1.  Extracts the lane edges from an image.
  Args:
    img: a NumPy array of shape (height, width, 3) storing an image
  Returns:
    A tuple of 2 boolean NumPy arrays each with shape (height, width), containing the edges of the left lane and the right lane respectively
  """
  ### Your code goes here! ###
  mask = get_bright_pixels(img)
  left_lane_begin, right_lane_begin = get_lane_beginnings(mask)
  left_lane_whole, right_lane_whole = get_whole_lanes(mask, left_lane_begin, right_lane_begin) # [[y, x], [y, x], ...]
  return get_lane_edges(left_lane_whole, right_lane_whole)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 3.1:  get_quadratic_fit 	#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

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
  ### Your code goes here! ###

  # Hint: generate an appropriate coordinate grid
  h, w = left_lane_edge.shape
  grid = np.indices((h, w)).transpose(1, 2, 0)

  # Hint: use NumPy slicing and indexing to pass the correct coordinates of edges to fit the polynomial
  left_poly = np.polynomial.polynomial.Polynomial.fit(x=grid[left_lane_edge][:, 0], y=grid[left_lane_edge][:, 1], deg=2)
  right_poly = np.polynomial.polynomial.Polynomial.fit(x=grid[right_lane_edge][:, 0], y=grid[right_lane_edge][:, 1], deg=2)

  return left_poly, right_poly


# No tasks to complete here.

def get_middle_quadratic(left_poly, right_poly):
  """ Returns the average of the left and right polynomial """
  # average left and right quadratic, and handle NumPy domain issues
  middle_poly = (left_poly.convert(domain=[0, 1439]) + right_poly.convert(domain=[0, 1439])) / 2
  a, b, c, d = left_poly.domain[0], right_poly.domain[0], left_poly.domain[1], right_poly.domain[1]
  middle_poly = middle_poly.convert(domain=[min(a, b, c, d), max(a, b, c, d)])
  return middle_poly


# No tasks to complete here.
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


# No tasks to complete here.

def get_direction_vector(curv, direction):
  """ Finds the direction vector to move in """
  r = 1 / curv # find radius of tangent circle
  s = 350 # move 350 pixels along the tangent circle
  if (direction == "left"):
    dx = - r + r * np.cos(s/r)
    dy = - r * np.sin(s/r)
  elif (direction == "right"):
    dx = + r - r * np.cos(s/r)
    dy = - r * np.sin(s/r)
  return dx, dy


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 3.2:  extract_direction_vector #-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def extract_direction_vector(img):
  """
  Finds how much to move in the x and y direction.
  Args:
    img: a NumPy array of shape (height, width, 3) storing an image
  Returns:
    a tuple containing 2 floating point numbers.  The first one is dx, the second one is dy.
  """
  ### TODO: Your code goes here! ###

  left_lane_edge, right_lane_edge = extract_lane_edges(img)
  left_poly, right_poly = get_quadratic_fit(left_lane_edge, right_lane_edge)
  middle_poly = get_middle_quadratic(left_poly, right_poly)
  curv, direction = get_curvature_and_direction(middle_poly)
  return get_direction_vector(curv, direction)



# Helper function.  No tasks to complete here.

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
  plt.scatter(x=left_x, y=left_y, color='green', s=7) # plot left poly
  plt.scatter(x=middle_x, y=middle_y, color='yellow', s=7) # plot middle poly
  plt.scatter(x=right_x, y=right_y, color='blue', s=7) # plot right poly
  plt.scatter(x=[middle_poly(pt)], y=[pt], color='purple') # plot point to evaluate curvature at
  plt.arrow(x=middle_poly(pt), y=pt, dx=dx, dy=dy, width=20, color=(1, 0, 0)) # draw direction vector
  plt.text(x=20, y=h-20, s=f"{direction} by {np.abs(dx)} px", fontsize='xx-large', color='white')

  plt.show()

if __name__ == '__main__':
  # Put the given img.png in your Colab files.
  # Your output should be the exactly same as the one below.
  if os.path.exists('img.png'):
    img = cv.imread('img.png')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    show_polynomials_and_vector(img)
  else:
    print("\"img.png\" not found in Colab files!")


# No tasks to complete here.

def video_to_np(filename):
  """
  Pass in the filename (a string), returns a tuple containing the frames and the dx values for those frames
  Usage example: x, y = video_to_np("drive/MyDrive/video1.MOV")
  """
  cap = cv.VideoCapture(filename)
  counter = 0
  x = [] # stores the frames
  y = [] # stores the dx's
  while (cap.isOpened()):
    ret, frame = cap.read() # 'ret' is a boolean, where True signifies the frame is read successfully. 'frame' is a NumPy array of the image.
    # break if reached end of video
    if (not ret):
      break
    # only consider every 30th frame
    if (counter % 30 == 0):
      # try to extract dx (it may not work if the video contains frames with unclear or non-existing road markings, don't worry about this)
      try:
        dx, _ = extract_direction_vector(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        y.append(dx)
      except Exception as e:
        # print(f"Error in frame {counter}:", e)
        y.append(y[-1] if len(y) > 0 else 0)
      frame = cv.resize(frame, (320, 240)) # reduce resolution of image.  cv.resize uses (width, height)
      frame = np.expand_dims(cv.cvtColor(frame, cv.COLOR_RGB2GRAY), axis=2)  # change to grayscale and reshape to (height, width, 1)
      frame[:120] = 0 # change the upper half of the image to black
      x.append(frame)
    # increment frame number counter
    counter += 1

  return x, y


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 4.1:  format_dataset #-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

def format_dataset(filenames, video_to_np):
  """
  Packs every 5 seconds of footage together, and also the mapped dx value corresponding to the last frame.
  Args:
    filenames: a list of strings.  For example ["drive/MyDrive/video1.MOV", "drive/MyDrive/video2.MOV"]
    video_to_np: this parameter exists so that we can test your code without your mistakes carrying over
  Returns:
    A tuple containing 2 NumPy arrays, the first one storing the frames and the second one storing the mapped dx values
  """
  ### Your code goes here! ###

  xs = [] # create a list to store the frames for each video
  ys = [] # create a list to store the dx values for each video

  for filename in filenames:
    if __name__ == "__main__":
      print(f"Processing {filename}")
    x, y = video_to_np(filename)

    # pack every consecutive 5 seconds into one training sample.
    # e.g. if frames are [A, B, C, D, E, F, G] this returns [[A, B, C, D, E], [B, C, D, E, F], [C, D, E, F, G]]
    x = np.expand_dims(np.array(x), 0)
    index = np.arange(0, 5).reshape(1, -1) + np.arange(0, x.shape[1]-4).reshape(-1, 1)
    xs.append(x[0, index])

    # handle y
    ys.append(y[4:])

  # map the dx values
  ys = np.concatenate(ys)
  zs = np.zeros_like(ys)
  zs[(ys >= -10) & (ys <= 10)] = 1
  zs[ys > 10] = 2

  return np.concatenate(xs), zs


def get_model():
    """
    Compiles and builds a branching CNN model for direction prediction
    with less than 10M trainable parameters.

    Returns:
      The compiled Keras model.
    """
    # Input: A sequence of 5 grayscale images of size 240x320.
    input_layer = layers.Input(shape=(5, 240, 320, 1))

    ### Branch 1: Spatial features extraction from the last frame ###
    # For this branch, use only the last frame.
    spatial = layers.Cropping3D(cropping=((4, 0), (0, 0), (0, 0)),
                                data_format="channels_last")(input_layer)
    # Reshape from (1, 240, 320, 1) to (240, 320, 1)
    spatial = layers.Reshape((240, 320, 1))(spatial)

    # Use a couple of Conv2D layers with fewer filters
    spatial = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding="same")(spatial)
    spatial = layers.Dropout(rate=0.3)(spatial)
    spatial = layers.MaxPooling2D(pool_size=(3, 3))(spatial)

    spatial = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same")(spatial)
    spatial = layers.Dropout(rate=0.3)(spatial)
    spatial = layers.MaxPooling2D(pool_size=(3, 3))(spatial)

    spatial = layers.Flatten()(spatial)
    spatial = layers.Dense(32, activation='relu')(spatial)

    ### Branch 2: Temporal and spatiotemporal feature extraction using 3D convolutions ###
    # Use the full 5-frame sequence.
    temporal = layers.Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', padding="same")(input_layer)
    temporal = layers.Dropout(rate=0.3)(temporal)
    temporal = layers.MaxPooling3D(pool_size=(1, 3, 3), padding="same")(temporal)

    temporal = layers.Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding="same")(temporal)
    temporal = layers.Dropout(rate=0.3)(temporal)
    temporal = layers.MaxPooling3D(pool_size=(1, 3, 3), padding="same")(temporal)

    temporal = layers.Flatten()(temporal)
    temporal = layers.Dense(16, activation='relu')(temporal)

    ### Merge branches ###
    merged = layers.concatenate([spatial, temporal])

    ### Dense layers ###
    x = layers.Dense(32, activation='relu')(merged)
    x = layers.Dropout(rate=0.1)(x)
    # Final classification over 3 classes.
    output = layers.Dense(3, activation='softmax')(x)

    # Model creation and compilation
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if __name__ == "__main__":
        model.summary()

    return model


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 4.3:  Train and test your model!   #-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

if __name__ == "__main__":

  ### TODO: Feel free to change the following code! ###

  # get the dataset
  filenames = ["/content/drive/MyDrive/HKUST_IA/2025S/COMP2211/PA2/video1.MOV",
               "/content/drive/MyDrive/HKUST_IA/2025S/COMP2211/PA2/video2.MOV",
               "/content/drive/MyDrive/HKUST_IA/2025S/COMP2211/PA2/video3.MOV",
               "/content/drive/MyDrive/HKUST_IA/2025S/COMP2211/PA2/video4.MOV"
               ]
  x, y = format_dataset(filenames, video_to_np)

  np.save("x.npy", x)
  np.save("y.npy", y)

"""Now, choose an appropriate number of epochs and batch size.  The code will help you save the trained model as `"COMP2211_PA2_Mini_AutoPilot.keras"`.  Refer to the course webpage for information on how to submit your code and model."""

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 4.3:  Train and test your model!   #-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
import keras
if __name__ == "__main__":
  x = np.load("x.npy")
  y = np.load("y.npy")
  train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=2211)

  # get the model
  if os.path.exists("COMP2211_PA2_Mini_AutoPilot.keras"):
    model = keras.saving.load_model("COMP2211_PA2_Mini_AutoPilot.keras")
  else:
    model = get_model()
  model.fit(x=train_x, y=train_y, epochs=20, batch_size=8, validation_data=(test_x, test_y))

  # test the model
  loss, accuracy = model.evaluate(x=test_x, y=test_y, batch_size=8)
  print(f"loss: {loss}:.3f")
  print(f"accuracy: {accuracy}:.3f")

  # save the model
  model.save("COMP2211_PA2_Mini_AutoPilot.keras")

"""The following code is provided for you to check the outputs of your model."""

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# TASK 4.3:  Train and test your model!   #-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

if __name__ == "__main__":
  # load model
  model = models.load_model("COMP2211_PA2_Mini_AutoPilot.keras")

  # choose some instance to check
  instances = [10, 11, 14, 18]

  for i in instances:
    # get the predicted and actual label
    text = lambda label: "left" if label == 0 else "straight" if label == 1 else "right"
    predicted = text(np.argmax(model(test_x[i:i+1]), axis=1))
    actual = text(int(test_y[i]))

    # display the image and results
    plt.imshow(test_x[i, 4])
    plt.show()
    print("predicted: ", predicted)
    print("actual:", actual)
    print("---")

"""##### The end

Congratulations!  You have reached the end of this assignment.  Even though a lot of work was done, there are still lots of ways to improve the results.

- The lane detection suffers from <b>camera angle</b> and <b>unclear road marking</b> issues.  Is there a way to map the image into an aerial view?
- A lot of <b>hyperparameters</b> were chosen, for example using 175 as a threshold for bright pixels (section 2.1), using 9 bounding boxes (section 2.3) and blackening the top half of the image (section 4.1).  Are there smarter ways to choose these hyperparameters?
- Our simplified model only predicts "left", "straight" or "right".  With more training data and better curvature detection methods, this could be improved.
- 3D convolutions were used to incorporate "time", but are there any better methods?
- The direction vector predicted by the model jumps around quite often.  Are there any ways to smooth out the direction vector?

Regardless, hope you were able to learn a lot from this assignment.  Remember to refer to the grading scheme on the course webpage, and submit your assignment in the correct format and on time.
"""