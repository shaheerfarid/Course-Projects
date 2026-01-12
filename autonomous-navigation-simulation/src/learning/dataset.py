import numpy as np

def format_dataset(filenames, video_to_np):
  """
  Packs every 5 seconds of footage together, and also the mapped dx value corresponding to the last frame.
  Args:
    filenames: a list of strings.  For example ["drive/MyDrive/video1.MOV", "drive/MyDrive/video2.MOV"]
    video_to_np: this parameter exists so that we can test your code without your mistakes carrying over
  Returns:
    A tuple containing 2 NumPy arrays, the first one storing the frames and the second one storing the mapped dx values
  """
 
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

