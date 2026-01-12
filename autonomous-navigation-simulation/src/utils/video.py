import numpy as np
import cv2 as cv
from control.steering import extract_direction_vector


def video_to_np(filename):
    """
    Pass in the filename (a string), returns a tuple containing the frames and the dx values for those frames
    Usage example: x, y = video_to_np("drive/MyDrive/video1.MOV")
    """
    cap = cv.VideoCapture(filename)
    counter = 0
    x = []  # stores the frames
    y = []  # stores the dx's
    while cap.isOpened():
        ret, frame = (
            cap.read()
        )  # 'ret' is a boolean, where True signifies the frame is read successfully. 'frame' is a NumPy array of the image.
        # break if reached end of video
        if not ret:
            break
        # only consider every 30th frame
        if counter % 30 == 0:
            # try to extract dx (it may not work if the video contains frames with unclear or non-existing road markings, don't worry about this)
            try:
                dx, _ = extract_direction_vector(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                y.append(dx)
            except Exception as e:
                # print(f"Error in frame {counter}:", e)
                y.append(y[-1] if len(y) > 0 else 0)
            frame = cv.resize(
                frame, (320, 240)
            )  # reduce resolution of image.  cv.resize uses (width, height)
            frame = np.expand_dims(
                cv.cvtColor(frame, cv.COLOR_RGB2GRAY), axis=2
            )  # change to grayscale and reshape to (height, width, 1)
            frame[:120] = 0  # change the upper half of the image to black
            x.append(frame)
        # increment frame number counter
        counter += 1

    return x, y
