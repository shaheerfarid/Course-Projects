import numpy as np
from keras import models
from sklearn.model_selection import train_test_split
import os
from learning.dataset import format_dataset
from learning.model import get_model
from utils import video_to_np
import matplotlib.pyplot as plt


if __name__ == "__main__":

  # get the dataset
  filenames = ["dataset/video1.MOV",
               "dataset/video2.MOV",
               "dataset/video3.MOV",
               "dataset/video4.MOV"
               ]
  x, y = format_dataset(filenames, video_to_np)

  np.save("x.npy", x)
  np.save("y.npy", y)

import keras
if __name__ == "__main__":
  x = np.load("x.npy")
  y = np.load("y.npy")
  train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=2211)

  # get the model
  if os.path.exists("autonomous-navigation-simulation.keras"):
    model = keras.saving.load_model("autonomous-navigation-simulation.keras")
  else:
    model = get_model()
  model.fit(x=train_x, y=train_y, epochs=20, batch_size=8, validation_data=(test_x, test_y))

  # test the model
  loss, accuracy = model.evaluate(x=test_x, y=test_y, batch_size=8)
  print(f"loss: {loss}:.3f")
  print(f"accuracy: {accuracy}:.3f")

  # save the model
  model.save("autonomous-navigation-simulation.keras")

"""The following code is provided to check the outputs of your model."""

if __name__ == "__main__":
  # load model
  model = models.load_model("autonomous-navigation-simulation.keras")

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